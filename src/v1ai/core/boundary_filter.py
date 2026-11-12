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
    Query-driven Multi-task MLP model predicting a total of 4 dimensions:
    "Presence of boundary (1D continuous confidence)" and "Line color RGB (3D)".
    Supports up to 10 colors, structured with brightness-sorted [R, G, B, flag] x 10 = 40 dimensions.
    """
    def __init__(self, config=None, patch_size=9, max_query_colors=10):
        super().__init__()
        self.patch_size = patch_size
        self.max_query_colors = max_query_colors  # Max 10 colors
        
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

        # MLP network structure: patch + query (40 dimensions: [R, G, B, flag] x 10)
        patch_dim = self.patch_size * self.patch_size * 3
        query_color_dim = self.max_query_colors * 4  # 40 dimensions
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
            nn.Linear(256, 4)  # Output: [is_edge (logit/sigmoided), R, G, B]
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_msg = (
            f"📊 [BoundaryFilter] Model Architecture Initialized (Optimized 4D Output Mode):\n"
            f" • Patch Size          : {self.patch_size}x{self.patch_size}\n"
            f" • Input Dimension     : {input_dim} (Patch: {patch_dim} + Query 40dim)\n"
            f" • Output Dimension    : 4 [Edge Score, R, G, B]\n"
            f" • Total Parameters    : {total_params:,}\n"
            f" • Trainable Parameters: {trainable_params:,}"
        )
        self._log(param_msg)

    def forward(self, patch_x, query_colors):
        """
        Args:
            patch_x: (B, 3, ps, ps)
            query_colors: (B, 40) ([R, G, B, flag] vectors for up to 10 colors, brightness sorted)
        Returns:
            torch.Tensor: (B, 4) -> [0]: Edge confidence [0, 1], [1:4]: RGB predicted values [0, 1]
        """
        b_size = patch_x.size(0)
        flat_patch = patch_x.view(b_size, -1)
        combined = torch.cat([flat_patch, query_colors], dim=1)
        out = self.net(combined)
        
        # Normalize edge score and RGB to [0, 1] using Sigmoid
        edge_score = torch.sigmoid(out[:, 0:1])
        rgb_pred = torch.sigmoid(out[:, 1:])
        return torch.cat([edge_score, rgb_pred], dim=1)

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
                
                label = y_data[data_idx][0]
                target_rgb = y_data[data_idx][1:4]

                ax.imshow(patch, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                
                rgb_hex = f"#{int(target_rgb[0]*255):02x}{int(target_rgb[1]*255):02x}{int(target_rgb[2]*255):02x}"
                ax.set_title(f"[EdgeScore: {label:.2f}] RGB: {rgb_hex}", fontsize=7)

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
        
        print(f"Generating {num_samples:,} training samples (Optimized Soft-Edge & Brightness-Sorted Query)...")
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
            shape_types = np.random.choice(['line', 'circle', 'rect', 'multi', 'smooth_grad'], size=current_batch_size, p=[0.25, 0.2, 0.2, 0.2, 0.15])
            
            num_useds = np.random.choice([2, 3, 4, 5], size=current_batch_size)
            noise_prob = 0.05
            
            for b in range(current_batch_size):
                num_used = num_useds[b]
                used_colors = np.random.choice(self.num_palette_colors, size=num_used, replace=False)
                st = shape_types[b]
                
                # Default grid generation
                if st == 'line':
                    theta = np.random.uniform(0, 2 * np.pi)
                    nx, ny = np.cos(theta), np.sin(theta)
                    norm = np.sqrt(nx**2 + ny**2) + 1e-8
                    dx, dy = nx / norm, ny / norm
                    offset = np.random.uniform(-0.6, 0.6)
                    
                    vals = dx * (x_indices - center_idx) + dy * (y_indices - center_idx) + offset
                    # Soft edge simulation using smoothstep / interpolation
                    if np.random.rand() < 0.4:
                        # Soft gradient blending across the boundary
                        alpha = np.clip((vals + 1.0) / 2.0, 0.0, 1.0)
                        c1 = palettes[b, used_colors[0]]
                        c2 = palettes[b, used_colors[1 % num_used]]
                        patch_grid = c1[None, None, :] * (1 - alpha[..., None]) + c2[None, None, :] * alpha[..., None]
                    else:
                        c_ids = np.where(vals >= 0, used_colors[1 % num_used], used_colors[0])
                        noise_mask = np.random.rand(ps, ps) < noise_prob
                        if np.any(noise_mask):
                            rand_noise_c = np.random.choice(used_colors, size=(ps, ps))
                            c_ids = np.where(noise_mask, rand_noise_c, c_ids)
                        patch_grid = palettes[b, c_ids]
                        
                elif st == 'circle':
                    cx = np.random.uniform(2, ps - 3)
                    cy = np.random.uniform(2, ps - 3)
                    r = np.random.uniform(1.5, ps / 2.2)
                    dists = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
                    c_ids = np.where(dists <= r, used_colors[1 % num_used], used_colors[0])
                    patch_grid = palettes[b, c_ids]
                    
                elif st == 'smooth_grad':
                    # Soft blurred gradient fill simulating real-world shadow/blur
                    dx = np.random.uniform(-1, 1)
                    dy = np.random.uniform(-1, 1)
                    vals = dx * (x_indices - center_idx) + dy * (y_indices - center_idx)
                    alpha = 1.0 / (1.0 + np.exp(-vals)) # Sigmoid smooth edge
                    c1 = palettes[b, used_colors[0]]
                    c2 = palettes[b, used_colors[1 % num_used]]
                    patch_grid = c1[None, None, :] * (1 - alpha[..., None]) + c2[None, None, :] * alpha[..., None]
                    
                else:
                    centers = [(np.random.uniform(0, ps), np.random.uniform(0, ps)) for _ in range(num_used)]
                    dists_stack = np.stack([np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2) for cx, cy in centers], axis=-1)
                    closest_idxs = np.argmin(dists_stack, axis=-1)
                    c_ids = used_colors[closest_idxs % num_used]
                    patch_grid = palettes[b, c_ids]

                # Add noise and blur optionally
                noise = np.random.normal(0, 0.03, size=(ps, ps, 3))
                patch_grid = np.clip(patch_grid + noise, 0.0, 1.0)
                
                # Extract center color & neighbors for ground truth computation
                center_rgb = patch_grid[center_idx, center_idx]
                
                neighbor_rgbs = []
                diff_count = 0
                for oy, ox in neighbor_offsets:
                    ny_pos, nx_pos = center_idx + oy, center_idx + ox
                    if 0 <= ny_pos < ps and 0 <= nx_pos < ps:
                        n_rgb = patch_grid[ny_pos, nx_pos]
                        neighbor_rgbs.append(n_rgb)
                        # Color distance to measure boundary degree
                        if np.mean(np.abs(n_rgb - center_rgb)) > 0.12:
                            diff_count += 1
                
                # Continuous label based on neighbor variation (0.0 to 1.0)
                edge_score = float(diff_count) / 8.0
                
                # Target line color (average of center and surrounding differing colors)
                if diff_count > 0:
                    differing = [n for n in neighbor_rgbs if np.mean(np.abs(n - center_rgb)) > 0.12]
                    edge_rgb = np.mean([center_rgb] + differing, axis=0)
                else:
                    edge_rgb = center_rgb

                # Construct 40-dimensional brightness-sorted query vector [R, G, B, flag] x 10
                query_colors_list = []
                # Pick some palette colors including present ones
                query_palette_ids = list(np.random.choice(self.num_palette_colors, size=min(np.random.randint(2, self.max_query_colors + 1), self.num_palette_colors), replace=False))
                
                for pid in query_palette_ids:
                    query_colors_list.append(palettes[b, pid])
                
                # Sort query colors by brightness (R+G+B) for invariant ordered representation
                query_colors_list.sort(key=lambda c: np.sum(c))
                
                query_vector = np.zeros(self.max_query_colors * 4, dtype=np.float32)
                for q_i, q_c in enumerate(query_colors_list[:self.max_query_colors]):
                    query_vector[q_i * 4 : q_i * 4 + 3] = q_c
                    query_vector[q_i * 4 + 3] = 1.0  # Existence flag = 1.0

                X_patches.append(np.transpose(patch_grid, (2, 0, 1)))
                X_queries.append(query_vector)
                
                target_vector = np.array([edge_score, edge_rgb[0], edge_rgb[1], edge_rgb[2]], dtype=np.float32)
                y_data.append(target_vector)

        gen_elapsed = time.time() - start_t
        print(f"✅ Optimized Training Data generation completed in {gen_elapsed:.2f}s ({num_samples:,} samples)")

        X_p_arr = np.array(X_patches, dtype=np.float32)
        X_q_arr = np.array(X_queries, dtype=np.float32)
        y_arr = np.array(y_data, dtype=np.float32)

        self._visualize_training_samples_grid(X_p_arr, X_q_arr, y_arr, num_display=10)

        return X_p_arr, X_q_arr, y_arr

    def train_model(self, epochs=45, batch_size=128, num_samples=200000, patience=12):
        self._log(f"Training Optimized 4D-output model with {num_samples:,} samples...")
        
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
        
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
        
        # Since edge score is now a continuous target [0, 1], use SmoothL1Loss / MSELoss for both edge and color
        edge_criterion = nn.SmoothL1Loss(beta=0.2)
        color_criterion = nn.SmoothL1Loss(beta=0.2)

        best_loss = float('inf')
        patience_counter = 0

        print(f"\n--- 🚀 {self.patch_size}x{self.patch_size} BoundaryFilter (Optimized 4D Output) Training Started ---")
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
                
                preds = self(b_x, b_q)  # (B, 4) -> [edge_score, R_pred, G_pred, B_pred]
                
                # Loss calculation with continuous edge score
                loss_edge = edge_criterion(preds[:, 0:1], b_y[:, 0:1])
                
                # Weight color loss effectively where edge exists (b_y[:, 0:1] > 0.1)
                edge_mask_weight = b_y[:, 0:1]
                loss_color = color_criterion(preds[:, 1:] * edge_mask_weight, b_y[:, 1:] * edge_mask_weight)
                
                loss = loss_edge + 0.8 * loss_color
                
                loss.backward()
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
                    
                    loss_edge = edge_criterion(preds[:, 0:1], b_y[:, 0:1])
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