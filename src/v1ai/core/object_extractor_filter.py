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
    Optimized ObjectExtractorFilter incorporating:
    - 40-dimensional query inputs (RGB + Flag x 5 object colors + RGB + Flag x 5 background colors)
    - Distance features (Min/Mean distances between center pixel and object/bg query groups)
    - Diverse synthetic textures (Perlin-like noise, smooth gradients, blurs) to bridge the real-world gap
    - Balanced over-erase penalty (weight = 0.5)
    - Fully integrated tqdm progress bars for data generation and model training.
    """
    def __init__(self, config=None, patch_size=9, max_object_colors=5, max_bg_colors=5):
        super().__init__()
        self.patch_size = patch_size
        self.max_object_colors = max_object_colors  # 5 colors (20 dimensions with flag)
        self.max_bg_colors = max_bg_colors          # 5 colors (20 dimensions with flag)
        
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

        patch_dim = self.patch_size * self.patch_size * 3          # 9x9x3 = 243
        object_query_dim = self.max_object_colors * 4             # 5 colors * 4 ([R,G,B,flag]) = 20 dimensions
        bg_query_dim = self.max_bg_colors * 4                     # 5 colors * 4 ([R,G,B,flag]) = 20 dimensions
        distance_feature_dim = 4                                  # [obj_min_dist, obj_mean_dist, bg_min_dist, bg_mean_dist]
        
        input_dim = patch_dim + object_query_dim + bg_query_dim + distance_feature_dim # 243 + 20 + 20 + 4 = 287 dimensions
        
        # MLP structure
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
            nn.Linear(128, 4)  # Output: [is_background logit, R, G, B]
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_msg = (
            f"📊 [ObjectExtractorFilter] Initialized (Optimized 40D Query + Distance Features):\n"
            f" • Patch Size          : {self.patch_size}x{self.patch_size}x3\n"
            f" • Object Query Dim    : {object_query_dim} (Max {self.max_object_colors} colors w/ flag)\n"
            f" • BG Query Dim        : {bg_query_dim} (Max {self.max_bg_colors} colors w/ flag)\n"
            f" • Distance Feat Dim   : {distance_feature_dim}\n"
            f" • Total Input Dim     : {input_dim}\n"
            f" • Output Dimension    : 4 [Is_Background Logit, R, G, B]\n"
            f" • Total Parameters    : {total_params:,}\n"
            f" • Trainable Parameters: {trainable_params:,}"
        )
        self._log(param_msg)

    def forward(self, patch_x, object_colors, bg_colors, distance_features):
        b_size = patch_x.size(0)
        flat_patch = patch_x.view(b_size, -1)
        combined = torch.cat([flat_patch, object_colors, bg_colors, distance_features], dim=1)
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
        ps = self.patch_size
        center_idx = ps // 2
        print(f"Generating {num_samples:,} training samples (Texture-rich & Distance Features integrated)...")
        start_t = time.time()
        
        chunk_size = 25000
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        
        X_patches_list, X_obj_q_list, X_bg_q_list, X_dist_list, y_data_list = [], [], [], [], []
        y_indices, x_indices = np.ogrid[:ps, :ps]
        
        pbar = tqdm(
            range(num_chunks),
            desc="Generating Data Chunks",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for _ in pbar:
            current_b_size = min(chunk_size, num_samples - sum(len(p) for p in X_patches_list))
            if current_b_size <= 0:
                break
                
            palettes = np.random.uniform(0.0, 1.0, size=(current_b_size, 10, 3))
            
            shape_types = np.random.choice(['multi_rect', 'multi_circle', 'smooth_grad', 'perlin_texture', 'mixed'], size=current_b_size, p=[0.25, 0.25, 0.2, 0.15, 0.15])
            color_id_grids = np.zeros((current_b_size, ps, ps), dtype=np.int32)
            
            bg_ids_base = np.arange(5, 10)
            obj_ids_base = np.arange(0, 5)
            
            for i in range(current_b_size):
                st = shape_types[i]
                base_bg = np.random.choice(bg_ids_base)
                color_id_grids[i] = base_bg
                
                if st == 'smooth_grad' or st == 'perlin_texture':
                    continue

                num_shapes = np.random.randint(2, 6)
                for _ in range(num_shapes):
                    obj_c = np.random.choice(obj_ids_base)
                    bg_c = np.random.choice(bg_ids_base)
                    chosen_c = np.random.choice([obj_c, bg_c, obj_c])
                    
                    shape_kind = np.random.choice(['rect', 'circle', 'tri'])
                    if shape_kind == 'rect':
                        rx1, ry1 = np.random.uniform(0, ps*0.7, 2)
                        rx2, ry2 = rx1 + np.random.uniform(ps*0.1, ps*0.5, 2)
                        mask = (rx1 <= x_indices) & (x_indices <= rx2) & (ry1 <= y_indices) & (y_indices <= ry2)
                    elif shape_kind == 'circle':
                        cx, cy = np.random.uniform(0, ps, 2)
                        rad = np.random.uniform(ps * 0.1, ps * 0.4)
                        mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= rad**2
                    else:
                        ax, ay = np.random.uniform(0, ps, 2)
                        mask = np.abs(x_indices - ax) + np.abs(y_indices - ay) <= np.random.uniform(ps*0.15, ps*0.35)
                        
                    color_id_grids[i] = np.where(mask, chosen_c, color_id_grids[i])

            batch_grids = np.zeros((current_b_size, ps, ps, 3), dtype=np.float32)
            for i in range(current_b_size):
                if shape_types[i] == 'smooth_grad':
                    dx = np.random.uniform(-1, 1)
                    dy = np.random.uniform(-1, 1)
                    vals = dx * (x_indices - center_idx) + dy * (y_indices - center_idx)
                    alpha = 1.0 / (1.0 + np.exp(-vals))
                    c1 = palettes[i, np.random.choice(obj_ids_base)]
                    c2 = palettes[i, np.random.choice(bg_ids_base)]
                    batch_grids[i] = c1[None, None, :] * (1 - alpha[..., None]) + c2[None, None, :] * alpha[..., None]
                elif shape_types[i] == 'perlin_texture':
                    freq = np.random.uniform(0.3, 0.8)
                    noise_pattern = np.sin(x_indices * freq) * np.cos(y_indices * freq)
                    obj_c = np.random.choice(obj_ids_base)
                    bg_c = np.random.choice(bg_ids_base)
                    mask = noise_pattern > 0.0
                    c_ids = np.where(mask, obj_c, bg_c)
                    batch_grids[i] = palettes[i, c_ids]
                else:
                    c_grid = color_id_grids[i]
                    batch_grids[i] = palettes[i, c_grid]

            noise = np.random.normal(0, 0.03, size=batch_grids.shape)
            batch_grids = np.clip(batch_grids + noise, 0.0, 1.0).astype(np.float32)
            batch_grids_t = np.transpose(batch_grids, (0, 3, 1, 2))

            center_colors = batch_grids[np.arange(current_b_size), center_idx, center_idx]
            
            obj_palettes = palettes[:, 0:5, :] 
            bg_palettes = palettes[:, 5:10, :] 
            
            center_expanded = center_colors[:, None, :] 
            obj_dists = np.linalg.norm(center_expanded - obj_palettes, axis=-1) 
            bg_dists = np.linalg.norm(center_expanded - bg_palettes, axis=-1)   
            
            obj_min_dist = np.min(obj_dists, axis=-1, keepdims=True)
            obj_mean_dist = np.mean(obj_dists, axis=-1, keepdims=True)
            bg_min_dist = np.min(bg_dists, axis=-1, keepdims=True)
            bg_mean_dist = np.mean(bg_dists, axis=-1, keepdims=True)
            
            distance_features = np.concatenate([obj_min_dist, obj_mean_dist, bg_min_dist, bg_mean_dist], axis=-1).astype(np.float32)

            is_bg = bg_min_dist.squeeze(-1) < obj_min_dist.squeeze(-1)
            labels = np.where(is_bg, 1.0, 0.0).astype(np.float32)

            obj_q_vectors = np.zeros((current_b_size, 20), dtype=np.float32)
            bg_q_vectors = np.zeros((current_b_size, 20), dtype=np.float32)
            
            for b in range(current_b_size):
                obj_list = [palettes[b, cid] for cid in range(5)]
                obj_list.sort(key=lambda c: np.sum(c))
                for idx, c in enumerate(obj_list):
                    obj_q_vectors[b, idx*4 : idx*4+3] = c
                    obj_q_vectors[b, idx*4+3] = 1.0
                
                bg_list = [palettes[b, cid] for cid in range(5, 10)]
                bg_list.sort(key=lambda c: np.sum(c))
                for idx, c in enumerate(bg_list):
                    bg_q_vectors[b, idx*4 : idx*4+3] = c
                    bg_q_vectors[b, idx*4+3] = 1.0

            target_vectors = np.stack([labels, center_colors[:, 0], center_colors[:, 1], center_colors[:, 2]], axis=-1)

            X_patches_list.append(batch_grids_t)
            X_obj_q_list.append(obj_q_vectors)
            X_bg_q_list.append(bg_q_vectors)
            X_dist_list.append(distance_features)
            y_data_list.append(target_vectors)

        X_p_arr = np.concatenate(X_patches_list, axis=0)
        X_oq_arr = np.concatenate(X_obj_q_list, axis=0)
        X_bgq_arr = np.concatenate(X_bg_q_list, axis=0)
        X_dist_arr = np.concatenate(X_dist_list, axis=0)
        y_arr = np.concatenate(y_data_list, axis=0)

        gen_elapsed = time.time() - start_t
        print(f"✅ Optimized data generation completed in {gen_elapsed:.2f}s ({num_samples:,} samples)")

        self._visualize_training_samples_grid(X_p_arr, y_arr, num_display=10)
        return X_p_arr, X_oq_arr, X_bgq_arr, X_dist_arr, y_arr

    def train_model(self, epochs=45, batch_size=128, num_samples=200000, patience=12):
        self._log(f"Training ObjectExtractorFilter with {num_samples:,} samples...")
        
        X_patches, X_obj_q, X_bg_q, X_dist, y_data = self._generate_training_data(num_samples)

        dataset = TensorDataset(
            torch.tensor(X_patches), 
            torch.tensor(X_obj_q), 
            torch.tensor(X_bg_q), 
            torch.tensor(X_dist),
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

        print(f"\n--- 🚀 ObjectExtractorFilter Optimized Training Started ---")
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
            for b_x, b_oq, b_bgq, b_dist, b_y in train_pbar:
                b_x, b_oq, b_bgq, b_dist, b_y = b_x.to(self.device), b_oq.to(self.device), b_bgq.to(self.device), b_dist.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                
                preds = self(b_x, b_oq, b_bgq, b_dist)  # (B, 4)
                
                loss_bg = bce_criterion(preds[:, 0:1], b_y[:, 0:1])
                
                is_object_mask = (b_y[:, 0:1] < 0.5).float()
                pred_bg_probs = torch.sigmoid(preds[:, 0:1])
                over_erase_penalty = torch.mean(is_object_mask * (pred_bg_probs ** 2))
                
                not_bg_mask = (1.0 - b_y[:, 0:1])
                loss_color = color_criterion(preds[:, 1:] * not_bg_mask, b_y[:, 1:] * not_bg_mask)
                
                loss = loss_bg + 0.8 * loss_color + 0.5 * over_erase_penalty
                
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
                for b_x, b_oq, b_bgq, b_dist, b_y in val_pbar:
                    b_x, b_oq, b_bgq, b_dist, b_y = b_x.to(self.device), b_oq.to(self.device), b_bgq.to(self.device), b_dist.to(self.device), b_y.to(self.device)
                    preds = self(b_x, b_oq, b_bgq, b_dist)
                    
                    loss_bg = bce_criterion(preds[:, 0:1], b_y[:, 0:1])
                    is_object_mask = (b_y[:, 0:1] < 0.5).float()
                    pred_bg_probs = torch.sigmoid(preds[:, 0:1])
                    over_erase_penalty = torch.mean(is_object_mask * (pred_bg_probs ** 2))
                    
                    not_bg_mask = (1.0 - b_y[:, 0:1])
                    loss_color = color_criterion(preds[:, 1:] * not_bg_mask, b_y[:, 1:] * not_bg_mask)
                    
                    loss = loss_bg + 0.8 * loss_color + 0.5 * over_erase_penalty
                    
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