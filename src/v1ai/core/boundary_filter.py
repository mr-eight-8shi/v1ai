import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class BoundaryFilter(nn.Module):
    """
    Multi-task MLP model predicting a total of 4 dimensions:
    "Presence of boundary (1D)" and "Line color RGB (3D)".
    Supports variable colors from 2 to 10 (zero-padded to a fixed 30-dimensional vector for max 10 colors).
    """
    def __init__(self, config=None, patch_size=9, max_query_colors=10):
        super().__init__()
        self.patch_size = patch_size
        self.max_query_colors = max_query_colors  # Max 10 colors (fixed 30 dimensions)
        
        if config is not None:
            self.patch_size = getattr(config, "patch_size", patch_size)
            self.config = config
            self.device = getattr(config, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.space_dir = getattr(config, "space_dir", "./")
            self.logger = getattr(config, "logger", None)
        else:
            self.config = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.space_dir = "./"
            self.logger = None

        self.num_palette_colors = 10
        self.model_dir = os.path.join(self.space_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        suffix = f"_{self.patch_size}x{self.patch_size}" if self.patch_size != 9 else ""
        self.model_path = os.path.join(self.model_dir, f"boundary_confidence_classifier{suffix}.pth")

        # MLP network structure (changed output dimension to 4: 1D edge presence + 3D RGB color)
        patch_dim = self.patch_size * self.patch_size * 3
        query_color_dim = self.max_query_colors * 3  # 30 dimensions
        input_dim = patch_dim + query_color_dim
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # Output: [is_edge (logit), R, G, B]
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_msg = (
            f"📊 [BoundaryFilter] Model Architecture Initialized (Multi-task 4D Output Mode):\n"
            f" • Patch Size          : {self.patch_size}x{self.patch_size}\n"
            f" • Input Dimension     : {input_dim}\n"
            f" • Output Dimension    : 4 [Edge Logit, R, G, B]\n"
            f" • Total Parameters    : {total_params:,}\n"
            f" • Trainable Parameters: {trainable_params:,}"
        )
        self._log(param_msg)

    def forward(self, patch_x, query_colors):
        """
        Args:
            patch_x: (B, 3, ps, ps)
            query_colors: (B, 30) (RGB vectors for up to 10 colors, missing parts zero-padded)
        Returns:
            torch.Tensor: (B, 4) -> [0]: Edge decision logit, [1:4]: RGB predicted values
        """
        b_size = patch_x.size(0)
        flat_patch = patch_x.view(b_size, -1)
        combined = torch.cat([flat_patch, query_colors], dim=1)
        out = self.net(combined)
        
        # Keep edge part as logits, normalize RGB part to [0, 1] using Sigmoid
        edge_logit = out[:, 0:1]
        rgb_pred = torch.sigmoid(out[:, 1:])
        return torch.cat([edge_logit, rgb_pred], dim=1)

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _visualize_training_samples_grid(self, X_patches, X_queries, y_data, num_display=10):
        try:
            fig, axes = plt.subplots(2, 5, figsize=(18, 7))
            axes = axes.flatten()
            
            indices = np.random.choice(len(X_patches), size=min(num_display, len(X_patches)), replace=False)
            
            for idx, data_idx in enumerate(indices):
                ax = axes[idx]
                patch = np.transpose(X_patches[data_idx], (1, 2, 0))
                patch = np.clip(patch, 0.0, 1.0)
                
                query = X_queries[data_idx]
                label = y_data[data_idx][0]
                target_rgb = y_data[data_idx][1:4]
                
                colors_in_query = []
                for c_i in range(self.max_query_colors):
                    rgb = query[c_i*3:(c_i+1)*3]
                    if np.sum(rgb) > 0.001:
                        colors_in_query.append(rgb)

                ax.imshow(patch, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                
                status_str = "TRUE" if label > 0.5 else "FALSE"
                rgb_hex = f"#{int(target_rgb[0]*255):02x}{int(target_rgb[1]*255):02x}{int(target_rgb[2]*255):02x}"
                ax.set_title(f"[{status_str}] LineRGB: {rgb_hex}", fontsize=7)

            plt.tight_layout()
            out_path = os.path.join(self.space_dir, "training_samples_visualization.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"🖼️ Training samples grid saved to: [{out_path}]")
        except Exception as e:
            print(f"⚠️ Failed to generate sample visualization grid: {e}")

    def _generate_training_data(self, num_samples):
        X_patches, X_queries, y_data = [], [], []
        ps = self.patch_size
        center_idx = ps // 2
        
        print(f"Generating {num_samples:,} training samples (4D Output: Edge + RGB, Vectorized)...")
        start_t = time.time()
        
        y_indices, x_indices = np.ogrid[:ps, :ps]
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        batch_gen_size = 2000
        num_batches = (num_samples + batch_gen_size - 1) // batch_gen_size
        
        pbar = tqdm(
            range(num_batches), 
            desc="Generating Data", 
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for b_idx in pbar:
            current_batch_size = min(batch_gen_size, num_samples - b_idx * batch_gen_size)
            
            palettes = np.random.uniform(0.0, 1.0, size=(current_batch_size, self.num_palette_colors, 3))
            shape_types = np.random.choice(['line', 'circle', 'rect', 'multi'], size=current_batch_size, p=[0.3, 0.2, 0.2, 0.3])
            color_id_grids = np.zeros((current_batch_size, ps, ps), dtype=np.int32)
            
            num_useds = np.random.choice([2, 3, 4, 5], size=current_batch_size)
            noise_prob = 0.05
            
            for b in range(current_batch_size):
                num_used = num_useds[b]
                used_colors = np.random.choice(self.num_palette_colors, size=num_used, replace=False)
                st = shape_types[b]
                
                if st == 'line':
                    theta = np.random.uniform(0, 2 * np.pi)
                    nx, ny = np.cos(theta), np.sin(theta)
                    norm = np.sqrt(nx**2 + ny**2) + 1e-8
                    dx, dy = nx / norm, ny / norm
                    offset = np.random.uniform(-0.6, 0.6)
                    
                    vals = dx * (x_indices - center_idx) + dy * (y_indices - center_idx) + offset
                    c_ids = np.where(vals >= 0, used_colors[1 % num_used], used_colors[0])
                    noise_mask = np.random.rand(ps, ps) < noise_prob
                    if np.any(noise_mask):
                        rand_noise_c = np.random.choice(used_colors, size=(ps, ps))
                        c_ids = np.where(noise_mask, rand_noise_c, c_ids)
                    color_id_grids[b] = c_ids
                    
                elif st == 'circle':
                    cx = np.random.uniform(2, ps - 3)
                    cy = np.random.uniform(2, ps - 3)
                    r = np.random.uniform(1.5, ps / 2.2)
                    dists = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
                    c_ids = np.where(dists <= r, used_colors[1 % num_used], used_colors[0])
                    noise_mask = np.random.rand(ps, ps) < noise_prob
                    if np.any(noise_mask):
                        rand_noise_c = np.random.choice(used_colors, size=(ps, ps))
                        c_ids = np.where(noise_mask, rand_noise_c, c_ids)
                    color_id_grids[b] = c_ids
                    
                elif st == 'rect':
                    rx1, ry1 = np.random.uniform(0, 2, 2)
                    rx2, ry2 = np.random.uniform(ps - 3, ps, 2)
                    inside = (rx1 <= x_indices) & (x_indices <= rx2) & (ry1 <= y_indices) & (y_indices <= ry2)
                    c_ids = np.where(inside, used_colors[1 % num_used], used_colors[0])
                    noise_mask = np.random.rand(ps, ps) < noise_prob
                    if np.any(noise_mask):
                        rand_noise_c = np.random.choice(used_colors, size=(ps, ps))
                        c_ids = np.where(noise_mask, rand_noise_c, c_ids)
                    color_id_grids[b] = c_ids
                    
                else:
                    centers = [(np.random.uniform(0, ps), np.random.uniform(0, ps)) for _ in range(num_used)]
                    dists_stack = np.stack([np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2) for cx, cy in centers], axis=-1)
                    closest_idxs = np.argmin(dists_stack, axis=-1)
                    color_id_grids[b] = used_colors[closest_idxs % num_used]

            grids = np.zeros((current_batch_size, ps, ps, 3), dtype=np.float32)
            for b in range(current_batch_size):
                grids[b] = palettes[b, color_id_grids[b]]

            noise = np.random.normal(0, 0.04, size=(current_batch_size, ps, ps, 3))
            grids = np.clip(grids + noise, 0.0, 1.0)
            grids_transposed = np.transpose(grids, (0, 3, 1, 2))

            for b in range(current_batch_size):
                c_grid = color_id_grids[b]
                center_color_id = c_grid[center_idx, center_idx]
                
                neighbor_colors = []
                for oy, ox in neighbor_offsets:
                    ny_pos, nx_pos = center_idx + oy, center_idx + ox
                    if 0 <= ny_pos < ps and 0 <= nx_pos < ps:
                        neighbor_colors.append(c_grid[ny_pos, nx_pos])
                
                different_count = sum(1 for n_id in neighbor_colors if n_id != center_color_id)
                is_on_boundary = (different_count >= 3)
                
                query_color_ids = []
                diff_neighbor_ids = [n_id for n_id in neighbor_colors if n_id != center_color_id]
                
                num_query_colors = np.random.randint(2, self.max_query_colors + 1)
                
                # Use boundary color (RGB) when on an edge as the correct color
                edge_rgb = palettes[b, center_color_id] # Default
                
                if is_on_boundary and np.random.rand() < 0.85 and len(diff_neighbor_ids) > 0:
                    boundary_neighbor_id = np.random.choice(diff_neighbor_ids)
                    query_color_ids.append(center_color_id)
                    query_color_ids.append(boundary_neighbor_id)
                    
                    # Target line color (e.g., average of center and neighbor colors, or one of them)
                    edge_rgb = (palettes[b, center_color_id] + palettes[b, boundary_neighbor_id]) / 2.0
                    
                    used_colors = np.unique(c_grid)
                    other_c = [c for c in used_colors if c != center_color_id and c != boundary_neighbor_id]
                    for oc in other_c[:num_query_colors - 2]:
                        query_color_ids.append(oc)
                    
                    while len(query_color_ids) < num_query_colors:
                        rand_c = np.random.randint(0, self.num_palette_colors)
                        if rand_c not in query_color_ids:
                            query_color_ids.append(rand_c)
                    label = 1.0
                else:
                    query_color_ids = list(np.random.choice(self.num_palette_colors, size=min(num_query_colors, self.num_palette_colors), replace=False))
                    while len(query_color_ids) < num_query_colors:
                        rand_c = np.random.randint(0, self.num_palette_colors)
                        if rand_c not in query_color_ids:
                            query_color_ids.append(rand_c)
                    
                    match_found = False
                    if is_on_boundary:
                        q_set = set(query_color_ids)
                        if center_color_id in q_set and any(n_id in q_set for n_id in neighbor_colors if n_id != center_color_id):
                            match_found = True
                    label = 1.0 if match_found else 0.0

                query_colors_list = [palettes[b, cid] for cid in query_color_ids[:num_query_colors]]
                np.random.shuffle(query_colors_list)
                
                query_vector = np.zeros(self.max_query_colors * 3, dtype=np.float32)
                flat_qc = np.concatenate(query_colors_list) if len(query_colors_list) > 0 else np.zeros(3)
                query_vector[:len(flat_qc)] = flat_qc

                X_patches.append(grids_transposed[b])
                X_queries.append(query_vector)
                
                # Teacher data: [label, R, G, B] (total 4 dimensions)
                target_vector = np.array([label, edge_rgb[0], edge_rgb[1], edge_rgb[2]], dtype=np.float32)
                y_data.append(target_vector)

        gen_elapsed = time.time() - start_t
        print(f"✅ Vectorized 4D Training Data generation completed in {gen_elapsed:.2f}s ({num_samples:,} samples)")

        X_p_arr = np.array(X_patches, dtype=np.float32)
        X_q_arr = np.array(X_queries, dtype=np.float32)
        y_arr = np.array(y_data, dtype=np.float32)

        self._visualize_training_samples_grid(X_p_arr, X_q_arr, y_arr, num_display=10)

        return X_p_arr, X_q_arr, y_arr

    def train_model(self, epochs=45, batch_size=128, num_samples=200000, patience=12):
        self._log(f"Training 4D-output model with {num_samples:,} samples...")
        
        X_patches, X_queries, y_data = self._generate_training_data(num_samples)

        dataset = TensorDataset(
            torch.tensor(X_patches), 
            torch.tensor(X_queries), 
            torch.tensor(y_data)
        )
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        self.to(self.device)
        
        # 1. Slightly adjust learning rate (0.0015 -> 0.001)
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
        
        # 2. Optimize loss function: consider pos_weight for edge detection, and use SmoothL1Loss robustly for colors
        # Adjust pos_weight according to the edge ratio (e.g., increase positive class weight by 1.8x since edges are sparse)
        pos_weight = torch.tensor([1.8], device=self.device)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Adopt SmoothL1Loss (Huber Loss), which is robust to outliers, instead of MSELoss
        color_criterion = nn.SmoothL1Loss(beta=0.2)

        best_loss = float('inf')
        patience_counter = 0

        print(f"\n--- 🚀 {self.patch_size}x{self.patch_size} BoundaryFilter (4D Output - Enhanced) Training Started ---")
        total_train_start = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            
            self.train()
            train_loss = 0.0
            
            train_pbar = tqdm(
                train_loader, 
                desc=f"Epoch [{epoch+1}/{epochs}] Train", 
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            for b_x, b_q, b_y in train_pbar:
                b_x, b_q, b_y = b_x.to(self.device), b_q.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                
                preds = self(b_x, b_q)  # (B, 4) -> [logit_edge, R_pred, G_pred, B_pred]
                
                # Loss calculation (edge presence + predicted color)
                loss_edge = bce_criterion(preds[:, 0:1], b_y[:, 0:1])
                
                # Make color loss strongly effective only on samples where "edge exists (y_data[:, 0] == 1)",
                # or take the average across the whole (here, apply a mask or weight to prioritize color learning on edges)
                edge_mask_weight = b_y[:, 0:1] # 1.0 only on edges, 0.0 (or negligible) on non-edges
                # To prevent loss from exploding when trying to force color matching on non-edge parts, prioritize color matching on edge parts
                loss_color = color_criterion(preds[:, 1:] * edge_mask_weight, b_y[:, 1:] * edge_mask_weight)
                
                # Combine multi-task losses (edge detection 1.0 vs color 0.8 balance)
                loss = loss_edge + 0.8 * loss_color
                
                loss.backward()
                # Gradient clipping to improve training stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * b_x.size(0)
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            avg_train_loss = train_loss / len(train_dataset)

            self.eval()
            val_loss = 0.0
            val_pbar = tqdm(
                val_loader, 
                desc=f"Epoch [{epoch+1}/{epochs}] Val  ", 
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            with torch.no_grad():
                for b_x, b_q, b_y in val_pbar:
                    b_x, b_q, b_y = b_x.to(self.device), b_q.to(self.device), b_y.to(self.device)
                    preds = self(b_x, b_q)
                    
                    loss_edge = bce_criterion(preds[:, 0:1], b_y[:, 0:1])
                    edge_mask_weight = b_y[:, 0:1]
                    loss_color = color_criterion(preds[:, 1:] * edge_mask_weight, b_y[:, 1:] * edge_mask_weight)
                    loss = loss_edge + 0.8 * loss_color
                    
                    val_loss += loss.item() * b_x.size(0)
                    val_pbar.set_postfix(val_loss=f"{loss.item():.4f}")

            avg_val_loss = val_loss / len(val_dataset)
            epoch_elapsed = time.time() - epoch_start

            print(f"👉 Epoch [{epoch+1}/{epochs}] ({epoch_elapsed:.1f}s) | Train Loss={avg_train_loss:.5f} | Val Loss={avg_val_loss:.5f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.state_dict(), self.model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1} (Best Val Loss: {best_loss:.5f})")
                    break

        total_elapsed = time.time() - total_train_start
        print(f"\n💾 Model saved to: [{self.model_path}] (Total Training Time: {total_elapsed/60:.2f} minutes)")