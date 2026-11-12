import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class ObjectExtractorFilter(nn.Module):
    """
    Multi-task MLP model predicting a total of 4 dimensions:
    "Whether it is background or not (1D: 1(False) if background, 0(True) if object/boundary)" and
    "RGB in case of non-background (3D)".
    
    [Query Specifications]
    - Utilizes 5 object colors (15D) + 5 background colors (15D) for a total of 10 colors (30D) for boundary judgment and features.
    - Object judgment mainly uses the "5 object colors".
    """
    def __init__(self, config=None, patch_size=9, max_object_colors=5, max_bg_colors=5):
        super().__init__()
        self.patch_size = patch_size
        self.max_object_colors = max_object_colors  # Representative object colors: 5 colors (15 dimensions)
        self.max_bg_colors = max_bg_colors          # Representative background colors: 5 colors (15 dimensions)
        
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

        self.model_dir = os.path.join(self.space_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        suffix = f"_{self.patch_size}x{self.patch_size}" if self.patch_size != 9 else ""
        self.model_path = os.path.join(self.model_dir, f"object_extractor_filter{suffix}.pth")

        patch_dim = self.patch_size * self.patch_size * 3     # 9x9x3 = 243
        object_query_dim = self.max_object_colors * 3         # 5 colors * 3 = 15 dimensions
        bg_query_dim = self.max_bg_colors * 3                 # 5 colors * 3 = 15 dimensions
        input_dim = patch_dim + object_query_dim + bg_query_dim # 243 + 15 + 15 = 273 dimensions
        
        # MLP structure (273 -> 512 -> 256 -> 128 -> 4)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # Output: [is_background logit (1/False if background), R, G, B]
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_msg = (
            f"📊 [ObjectExtractorFilter] Model Architecture Initialized (10-Color Total Mode):\n"
            f" • Patch Size          : {self.patch_size}x{self.patch_size}x3\n"
            f" • Object Query Dim    : {object_query_dim} (Max {self.max_object_colors} colors)\n"
            f" • BG Query Dim        : {bg_query_dim} (Max {self.max_bg_colors} colors)\n"
            f" • Input Dimension     : {input_dim}\n"
            f" • Output Dimension    : 4 [Is_Background Logit, R, G, B]\n"
            f" • Total Parameters    : {total_params:,}\n"
            f" • Trainable Parameters: {trainable_params:,}"
        )
        self._log(param_msg)

    def forward(self, patch_x, object_colors, bg_colors):
        b_size = patch_x.size(0)
        flat_patch = patch_x.view(b_size, -1)
        combined = torch.cat([flat_patch, object_colors, bg_colors], dim=1)
        out = self.net(combined)
        
        is_bg_logit = out[:, 0:1]
        rgb_pred = torch.sigmoid(out[:, 1:])
        return torch.cat([is_bg_logit, rgb_pred], dim=1)

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _visualize_training_samples_grid(self, X_patches, y_data, num_display=10):
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
                
                # Display purely the generated patch images without debug circles
                ax.imshow(patch, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                
                status_str = "BACKGROUND (False)" if label > 0.5 else "OBJECT/BOUNDARY (True)"
                rgb_hex = f"#{int(target_rgb[0]*255):02x}{int(target_rgb[1]*255):02x}{int(target_rgb[2]*255):02x}"
                ax.set_title(f"[{status_str}]\nRGB: {rgb_hex}", fontsize=6)

            plt.tight_layout()
            out_path = os.path.join(self.space_dir, "object_extractor_training_samples_visualization.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"🖼️ Training samples grid saved to: [{out_path}]")
        except Exception as e:
            print(f"⚠️ Failed to generate sample visualization grid: {e}")

    def _generate_training_data(self, num_samples):
        """
        Data generation mixing diverse shapes (rectangles, circles, triangles, random shapes) within the image,
        and using a logical center circle mask to separate "inside circle = 5 object colors" and "outside circle = 5 background colors".
        """
        ps = self.patch_size
        center_idx = ps // 2
        print(f"Generating {num_samples:,} training samples (Diverse Shapes & Logical 10-Color Split)...")
        start_t = time.time()
        
        chunk_size = 50000
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        
        X_patches_list, X_obj_q_list, X_bg_q_list, y_data_list = [], [], [], []
        y_indices, x_indices = np.ogrid[:ps, :ps]
        
        for _ in range(num_chunks):
            current_b_size = min(chunk_size, num_samples - len(X_patches_list))
            if current_b_size <= 0:
                break
                
            # 10-color palette [0-4: 5 object colors, 5-9: 5 background colors]
            palettes = np.random.uniform(0.0, 1.0, size=(current_b_size, 10, 3))
            
            # Mix diverse shapes (rectangles, large/small circles, triangle-like, lines, random polygons, etc.)
            shape_types = np.random.choice(['multi_rect', 'multi_circle', 'triangles', 'lines', 'mixed'], size=current_b_size, p=[0.25, 0.25, 0.2, 0.15, 0.15])
            color_id_grids = np.zeros((current_b_size, ps, ps), dtype=np.int32)
            
            bg_ids_base = np.arange(5, 10)
            obj_ids_base = np.arange(0, 5)
            
            for i in range(current_b_size):
                st = shape_types[i]
                # Set initial background color as base
                base_bg = np.random.choice(bg_ids_base)
                color_id_grids[i] = base_bg
                
                # Place multiple figures of different sizes and shapes (mixture of objects and background) in the image
                num_shapes = np.random.randint(2, 6)
                for _ in range(num_shapes):
                    obj_c = np.random.choice(obj_ids_base)
                    bg_c = np.random.choice(bg_ids_base)
                    chosen_c = np.random.choice([obj_c, bg_c, obj_c]) # Object colors slightly more frequent
                    
                    shape_kind = np.random.choice(['rect', 'circle', 'tri'])
                    if shape_kind == 'rect':
                        rx1, ry1 = np.random.uniform(0, ps*0.7, 2)
                        rx2, ry2 = rx1 + np.random.uniform(ps*0.1, ps*0.5, 2)
                        mask = (rx1 <= x_indices) & (x_indices <= rx2) & (ry1 <= y_indices) & (y_indices <= ry2)
                    elif shape_kind == 'circle':
                        cx, cy = np.random.uniform(0, ps, 2)
                        rad = np.random.uniform(ps * 0.1, ps * 0.4)
                        mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= rad**2
                    else: # Triangle-like (simple mask)
                        ax, ay = np.random.uniform(0, ps, 2)
                        mask = np.abs(x_indices - ax) + np.abs(y_indices - ay) <= np.random.uniform(ps*0.15, ps*0.35)
                        
                    color_id_grids[i] = np.where(mask, chosen_c, color_id_grids[i])

            # Apply noise
            noise_mask = np.random.rand(current_b_size, ps, ps) < 0.08
            random_noise_ids = np.random.choice(np.arange(10), size=(current_b_size, ps, ps))
            color_id_grids = np.where(noise_mask, random_noise_ids, color_id_grids)

            # Convert from palette to RGB
            batch_grids = np.take_along_axis(palettes[:, None, None, :, :], color_id_grids[..., None], axis=3).squeeze(3)
            noise = np.random.normal(0, 0.02, size=batch_grids.shape)
            batch_grids = np.clip(batch_grids + noise, 0.0, 1.0).astype(np.float32)
            batch_grids_t = np.transpose(batch_grids, (0, 3, 1, 2))

            center_cids = color_id_grids[:, center_idx, center_idx]
            center_colors = palettes[np.arange(current_b_size), center_cids]
            
            # Boundary judgment
            neighbors = []
            for oy in [-1, 0, 1]:
                for ox in [-1, 0, 1]:
                    if oy == 0 and ox == 0:
                        continue
                    neighbors.append(color_id_grids[:, center_idx+oy, center_idx+ox])
            neighbors_stack = np.stack(neighbors, axis=-1)
            diff_count = np.sum(neighbors_stack != center_cids[..., None], axis=-1)
            is_boundary = diff_count >= 3
            
            is_bg = center_cids >= 5  # 5 to 9 are background
            labels = np.where(is_boundary | (~is_bg), 0.0, 1.0).astype(np.float32)

            # Create query vector (first 5 colors are object side, latter 5 colors are background side)
            obj_vectors = palettes[:, 0:5, :].reshape(current_b_size, -1).astype(np.float32)
            bg_vectors = palettes[:, 5:10, :].reshape(current_b_size, -1).astype(np.float32)
            
            target_vectors = np.stack([labels, center_colors[:, 0], center_colors[:, 1], center_colors[:, 2]], axis=-1)

            X_patches_list.append(batch_grids_t)
            X_obj_q_list.append(obj_vectors)
            X_bg_q_list.append(bg_vectors)
            y_data_list.append(target_vectors)

        X_p_arr = np.concatenate(X_patches_list, axis=0)
        X_oq_arr = np.concatenate(X_obj_q_list, axis=0)
        X_bgq_arr = np.concatenate(X_bg_q_list, axis=0)
        y_arr = np.concatenate(y_data_list, axis=0)

        gen_elapsed = time.time() - start_t
        print(f"✅ Vectorized data generation completed in {gen_elapsed:.2f}s ({num_samples:,} samples)")

        self._visualize_training_samples_grid(X_p_arr, y_arr, num_display=10)
        return X_p_arr, X_oq_arr, X_bgq_arr, y_arr

    def _generate_training_data_legacy(self, num_samples):
        X_patches, X_obj_queries, X_bg_queries, y_data = [], [], [], []
        ps = self.patch_size
        center_idx = ps // 2
        
        print(f"Generating {num_samples:,} training samples for ObjectExtractorFilter (Diverse Shapes & 10 Colors)...")
        start_t = time.time()
        
        y_indices, x_indices = np.ogrid[:ps, :ps]
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        batch_gen_size = 2000
        num_batches = (num_samples + batch_gen_size - 1) // batch_gen_size
        
        pbar = tqdm(
            range(num_batches), 
            desc="Generating Object Extractor Data", 
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for b_idx in pbar:
            current_batch_size = min(batch_gen_size, num_samples - b_idx * batch_gen_size)
            
            # Generate palettes for 10 colors [0-4: 5 object colors, 5-9: 5 background colors]
            master_palettes = np.random.uniform(0.0, 1.0, size=(current_batch_size, 10, 3))
            
            # Diverse shape types (circle, rect, line, multi, solid)
            shape_types = np.random.choice(['circle', 'rect', 'line', 'multi', 'solid'], size=current_batch_size, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            
            color_id_grids = np.zeros((current_batch_size, ps, ps), dtype=np.int32)
            batch_obj_ids = []
            batch_bg_ids = []
            
            for b in range(current_batch_size):
                obj_ids = np.arange(0, 5)
                bg_ids = np.arange(5, 10)
                batch_obj_ids.append(obj_ids)
                batch_bg_ids.append(bg_ids)

                st = shape_types[b]
                
                if st == 'solid':
                    color_id_grids[b] = np.random.choice(bg_ids)
                elif st == 'circle':
                    # Place circles of various sizes and positions within the patch
                    cx = np.random.uniform(0, ps - 1)
                    cy = np.random.uniform(0, ps - 1)
                    r = np.random.uniform(ps * 0.15, ps * 0.6)
                    dists = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
                    bg_c = np.random.choice(bg_ids)
                    obj_c = np.random.choice(obj_ids)
                    color_id_grids[b] = np.where(dists <= r, obj_c, bg_c)
                elif st == 'rect':
                    # Rectangle placement
                    rx1, ry1 = np.random.uniform(0, ps * 0.4, 2)
                    rx2, ry2 = np.random.uniform(ps * 0.6, ps, 2)
                    inside = (rx1 <= x_indices) & (x_indices <= rx2) & (ry1 <= y_indices) & (y_indices <= ry2)
                    bg_c = np.random.choice(bg_ids)
                    obj_c = np.random.choice(obj_ids)
                    color_id_grids[b] = np.where(inside, obj_c, bg_c)
                elif st == 'line':
                    # Line / boundary placement
                    theta = np.random.uniform(0, 2 * np.pi)
                    nx, ny = np.cos(theta), np.sin(theta)
                    norm = np.sqrt(nx**2 + ny**2) + 1e-8
                    dx, dy = nx / norm, ny / norm
                    offset = np.random.uniform(-ps*0.3, ps*0.3)
                    vals = dx * (x_indices - center_idx) + dy * (y_indices - center_idx) + offset
                    bg_c = np.random.choice(bg_ids)
                    obj_c = np.random.choice(obj_ids)
                    color_id_grids[b] = np.where(vals >= 0, obj_c, bg_c)
                else:
                    # Multiple objects (Voronoi-like / multiple centers)
                    centers = [(np.random.uniform(0, ps), np.random.uniform(0, ps)) for _ in range(3)]
                    dists_stack = np.stack([np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2) for cx, cy in centers], axis=-1)
                    closest_idxs = np.argmin(dists_stack, axis=-1)
                    chosen_obj = np.random.choice(obj_ids)
                    mapping = np.array([np.random.choice(bg_ids), chosen_obj, chosen_obj])
                    color_id_grids[b] = mapping[closest_idxs]

                # Apply noise
                noise_mask = np.random.rand(ps, ps) < 0.08
                if np.any(noise_mask):
                    random_noise_ids = np.random.choice(np.arange(10), size=(ps, ps))
                    color_id_grids[b] = np.where(noise_mask, random_noise_ids, color_id_grids[b])

            grids = np.zeros((current_batch_size, ps, ps, 3), dtype=np.float32)
            for b in range(current_batch_size):
                palette = master_palettes[b]
                c_grid = color_id_grids[b]
                for y in range(ps):
                    for x in range(ps):
                        cid = c_grid[y, x]
                        grids[b, y, x] = palette[cid]

            noise = np.random.normal(0, 0.03, size=(current_batch_size, ps, ps, 3))
            grids = np.clip(grids + noise, 0.0, 1.0)
            grids_transposed = np.transpose(grids, (0, 3, 1, 2))

            for b in range(current_batch_size):
                c_grid = color_id_grids[b]
                palette = master_palettes[b]
                obj_ids = batch_obj_ids[b]
                bg_ids = batch_bg_ids[b]
                
                center_cid = c_grid[center_idx, center_idx]
                center_color = palette[center_cid]
                
                neighbor_cids = []
                for oy, ox in neighbor_offsets:
                    ny_pos, nx_pos = center_idx + oy, center_idx + ox
                    if 0 <= ny_pos < ps and 0 <= nx_pos < ps:
                        neighbor_cids.append(c_grid[ny_pos, nx_pos])
                
                different_count = sum(1 for n_id in neighbor_cids if n_id != center_cid)
                is_geometry_boundary = (different_count >= 3)
                is_center_bg = (center_cid in bg_ids)

                if is_geometry_boundary:
                    label = 0.0  # Boundary is True
                elif is_center_bg:
                    label = 1.0  # Background is False
                else:
                    label = 0.0  # Object region is True

                target_rgb = center_color

                # Extract 5 object colors and 5 background colors from each image and vectorize (15D + 15D)
                obj_colors_list = [palette[cid] for cid in obj_ids]
                bg_colors_list = [palette[cid] for cid in bg_ids]

                obj_vector = np.concatenate(obj_colors_list)
                bg_vector = np.concatenate(bg_colors_list)

                X_patches.append(grids_transposed[b])
                X_obj_queries.append(obj_vector)
                X_bg_queries.append(bg_vector)
                
                target_vector = np.array([label, target_rgb[0], target_rgb[1], target_rgb[2]], dtype=np.float32)
                y_data.append(target_vector)

        gen_elapsed = time.time() - start_t
        print(f"✅ Data generation completed in {gen_elapsed:.2f}s ({num_samples:,} samples)")

        X_p_arr = np.array(X_patches, dtype=np.float32)
        X_oq_arr = np.array(X_obj_queries, dtype=np.float32)
        X_bgq_arr = np.array(X_bg_queries, dtype=np.float32)
        y_arr = np.array(y_data, dtype=np.float32)

        self._visualize_training_samples_grid(X_p_arr, y_arr, num_display=10)

        return X_p_arr, X_oq_arr, X_bgq_arr, y_arr

    def train_model(self, epochs=45, batch_size=128, num_samples=200000, patience=12):
        self._log(f"Training ObjectExtractorFilter with {num_samples:,} samples...")
        
        X_patches, X_obj_q, X_bg_q, y_data = self._generate_training_data(num_samples)

        dataset = TensorDataset(
            torch.tensor(X_patches), 
            torch.tensor(X_obj_q), 
            torch.tensor(X_bg_q), 
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
        
        pos_weight = torch.tensor([1.2], device=self.device)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        color_criterion = nn.SmoothL1Loss(beta=0.2)

        best_loss = float('inf')
        patience_counter = 0

        print(f"\n--- 🚀 ObjectExtractorFilter Training Started ---")
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
            for b_x, b_oq, b_bgq, b_y in train_pbar:
                b_x, b_oq, b_bgq, b_y = b_x.to(self.device), b_oq.to(self.device), b_bgq.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                
                preds = self(b_x, b_oq, b_bgq) # (B, 4)
                
                loss_bg = bce_criterion(preds[:, 0:1], b_y[:, 0:1])
                
                is_object_mask = (b_y[:, 0:1] < 0.5).float()
                pred_bg_probs = torch.sigmoid(preds[:, 0:1])
                over_erase_penalty = torch.mean(is_object_mask * (pred_bg_probs ** 2))
                
                not_bg_mask = (1.0 - b_y[:, 0:1])
                loss_color = color_criterion(preds[:, 1:] * not_bg_mask, b_y[:, 1:] * not_bg_mask)
                
                loss = loss_bg + 0.8 * loss_color + 2.0 * over_erase_penalty
                
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
                for b_x, b_oq, b_bgq, b_y in val_pbar:
                    b_x, b_oq, b_bgq, b_y = b_x.to(self.device), b_oq.to(self.device), b_bgq.to(self.device), b_y.to(self.device)
                    preds = self(b_x, b_oq, b_bgq)
                    
                    loss_bg = bce_criterion(preds[:, 0:1], b_y[:, 0:1])
                    is_object_mask = (b_y[:, 0:1] < 0.5).float()
                    pred_bg_probs = torch.sigmoid(preds[:, 0:1])
                    over_erase_penalty = torch.mean(is_object_mask * (pred_bg_probs ** 2))
                    
                    not_bg_mask = (1.0 - b_y[:, 0:1])
                    loss_color = color_criterion(preds[:, 1:] * not_bg_mask, b_y[:, 1:] * not_bg_mask)
                    
                    loss = loss_bg + 0.8 * loss_color + 2.0 * over_erase_penalty
                    
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