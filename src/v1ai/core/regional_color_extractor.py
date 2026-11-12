import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class RegionalColorExtractor(nn.Module):
    """
    Optimized RegionalColorExtractor:
    - Separates RGB prediction (Sigmoid -> [0, 1]) and Existence Flag (Logits for BCEWithLogitsLoss).
    - Applies flag-masked RGB loss so non-existing colors do not incur useless penalties.
    - Uses advanced texture-mixed data generation (gradients, blurs, noise) to bridge the real-world gap.
    """
    def __init__(self, config=None, target_size=32):
        super().__init__()
        self.target_size = target_size
        
        if config is not None:
            self.target_size = getattr(config, "target_size", target_size)
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
        self.model_path = os.path.join(self.model_dir, "regional_color_extractor.pth")

        # Lightweight CNN feature extraction backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
            nn.Flatten()
        )
        
        # 40-dimensional output split:
        # First 30 dims: RGB for 10 colors (will be Sigmoided)
        # Latter 10 dims: Existence logits for 10 colors (will use BCEWithLogitsLoss)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 40)  # No final Sigmoid here; applied selectively in forward
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_msg = (
            f"📊 [RegionalColorExtractor v3] Initialized (Separated RGB & Flag Logits):\n"
            f" • Target Size         : {self.target_size}x{self.target_size}\n"
            f" • Total Parameters    : {total_params:,}\n"
            f" • Trainable Parameters: {trainable_params:,}"
        )
        self._log(param_msg)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (B, 3, H, W) or (B, H, W, 3), value range [0.0, 1.0]
        Returns:
            torch.Tensor: shape (B, 10, 4) -> RGB normalized to [0,1], Flag as logits
        """
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
            
        feat = self.features(x)
        out_flat = self.fc(feat)  # (B, 40)
        
        # Separate RGB (first 30 dims -> 10 * 3) and Flags (latter 10 dims -> 10 * 1)
        rgb_logits = out_flat[:, :30].view(-1, 10, 3)
        flag_logits = out_flat[:, 30:].view(-1, 10, 1)
        
        rgb_pred = torch.sigmoid(rgb_logits)
        
        # Concatenate back to (B, 10, 4) where channel 3 is flag logit
        out = torch.cat([rgb_pred, flag_logits], dim=2)
        return out

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _visualize_training_samples_grid(self, X_images, y_colors, num_display=10):
        try:
            fig, axes = plt.subplots(2, 5, figsize=(18, 7))
            axes = axes.flatten()
            
            indices = np.random.choice(len(X_images), size=min(num_display, len(X_images)), replace=False)
            
            for idx, data_idx in enumerate(indices):
                ax = axes[idx]
                img = np.transpose(X_images[data_idx], (1, 2, 0))
                img = np.clip(img, 0.0, 1.0)
                
                colors_40 = y_colors[data_idx]  # (40,)
                colors_10x4 = colors_40.reshape(10, 4)
                
                inside_items = colors_10x4[:5]   # Inner region colors
                outside_items = colors_10x4[5:]  # Outer region colors
                
                inside_hexes = ", ".join([f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}({c[3]:.1f})" for c in inside_items if c[3] > 0.5])
                outside_hexes = ", ".join([f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}({c[3]:.1f})" for c in outside_items if c[3] > 0.5])
                if not inside_hexes: inside_hexes = "None"
                if not outside_hexes: outside_hexes = "None"

                ax.imshow(img, interpolation='nearest')
                
                ts = self.target_size
                cy, cx = ts / 2.0, ts / 2.0
                radius = ts / 2.0
                theta = np.linspace(0, 2 * np.pi, 100)
                circle_y = cy + radius * np.sin(theta)
                circle_x = cx + radius * np.cos(theta)
                ax.plot(circle_x, circle_y, color='yellow', linestyle='--', linewidth=1.0, alpha=0.7)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Inner: {inside_hexes}\nOuter: {outside_hexes}", fontsize=5)

            plt.tight_layout()
            out_path = os.path.join(self.space_dir, "regional_extractor_32x32_v3_samples.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"🖼️ Training samples grid saved to: [{out_path}]")
        except Exception as e:
            print(f"⚠️ Failed to generate sample visualization grid: {e}")

    def _generate_training_data(self, num_samples):
        X_images, y_colors = [], []
        ts = self.target_size
        center = ts / 2.0
        
        print(f"Generating {num_samples:,} training samples ({ts}x{ts}) [v3: Gradients, Blurs & Masked RGB Loss]...")
        start_t = time.time()
        
        y_indices, x_indices = np.ogrid[:ts, :ts]
        batch_gen_size = 2000
        num_batches = (num_samples + batch_gen_size - 1) // batch_gen_size
        
        pbar = tqdm(
            range(num_batches), 
            desc="Generating Data (v3)", 
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for b_idx in pbar:
            current_batch_size = min(batch_gen_size, num_samples - b_idx * batch_gen_size)
            palettes = np.random.uniform(0.1, 0.9, size=(current_batch_size, 10, 3))
            
            radius_limit = ts / 2.0
            radius_mask = np.sqrt((x_indices - center)**2 + (y_indices - center)**2) <= radius_limit

            batch_grids = np.zeros((current_batch_size, ts, ts, 3), dtype=np.float32)
            color_id_grids = np.zeros((current_batch_size, ts, ts), dtype=np.int32)
            
            shape_types = np.random.choice(['shapes', 'smooth_grad', 'perlin_noise'], size=current_batch_size, p=[0.6, 0.2, 0.2])
            
            for b in range(current_batch_size):
                st = shape_types[b]
                base_bg_id = np.random.choice([5, 6, 7, 8, 9])
                color_id_grids[b] = base_bg_id
                
                if st == 'smooth_grad':
                    dx = np.random.uniform(-1, 1)
                    dy = np.random.uniform(-1, 1)
                    vals = dx * (x_indices - center) + dy * (y_indices - center)
                    alpha = 1.0 / (1.0 + np.exp(-vals))
                    c1 = palettes[b, np.random.choice(range(5))]
                    c2 = palettes[b, np.random.choice(range(5, 10))]
                    batch_grids[b] = c1[None, None, :] * (1 - alpha[..., None]) + c2[None, None, :] * alpha[..., None]
                    continue
                elif st == 'perlin_noise':
                    freq = np.random.uniform(0.4, 0.9)
                    pattern = np.sin(x_indices * freq) * np.cos(y_indices * freq)
                    mask = pattern > 0.0
                    obj_c = np.random.choice(range(5))
                    bg_c = np.random.choice(range(5, 10))
                    color_id_grids[b] = np.where(mask, obj_c, bg_c)
                else:
                    num_shapes = np.random.randint(1, 4)
                    for _ in range(num_shapes):
                        obj_id = np.random.choice([0, 1, 2, 3, 4])
                        bg_id = np.random.choice([5, 6, 7, 8, 9])
                        chosen_id = np.random.choice([obj_id, bg_id, obj_id])
                        
                        shape_kind = np.random.choice(['rect', 'circle', 'tri'])
                        if shape_kind == 'rect':
                            rx1, ry1 = np.random.uniform(0, ts*0.7, 2)
                            rx2, ry2 = rx1 + np.random.uniform(ts*0.1, ts*0.5, 2)
                            mask = (rx1 <= x_indices) & (x_indices <= rx2) & (ry1 <= y_indices) & (y_indices <= ry2)
                        elif shape_kind == 'circle':
                            cx, cy = np.random.uniform(0, ts, 2)
                            rad = np.random.uniform(ts * 0.15, ts * 0.4)
                            mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= rad**2
                        else:
                            ax, ay = np.random.uniform(0, ts, 2)
                            mask = np.abs(x_indices - ax) + np.abs(y_indices - ay) <= np.random.uniform(ts*0.15, ts*0.35)
                            
                        color_id_grids[b] = np.where(mask, chosen_id, color_id_grids[b])

            for b in range(current_batch_size):
                if shape_types[b] != 'smooth_grad':
                    palette = palettes[b]
                    c_grid = color_id_grids[b]
                    grid_rgb = palette[c_grid]
                    pixel_noise = np.random.normal(0, 0.03, size=(ts, ts, 3))
                    batch_grids[b] = np.clip(grid_rgb + pixel_noise, 0.0, 1.0)

            total_noise = np.random.normal(0, 0.02, size=(current_batch_size, ts, ts, 3))
            batch_grids = np.clip(batch_grids + total_noise, 0.0, 1.0)
            batch_grids_transposed = np.transpose(batch_grids, (0, 3, 1, 2))
            
            targets_list = []
            for b in range(current_batch_size):
                palette = palettes[b]
                c_grid = color_id_grids[b]
                in_mask = radius_mask
                out_mask = ~radius_mask
                
                # Inner 5 colors (RGB + existence flag 1.0 or 0.0)
                inside_features = []
                for cid in range(5):
                    match_pixels = (c_grid == cid) & in_mask
                    if np.any(match_pixels):
                        mean_col = np.mean(batch_grids[b][match_pixels], axis=0)
                        inside_features.append([mean_col[0], mean_col[1], mean_col[2], 1.0])
                    else:
                        inside_features.append([0.0, 0.0, 0.0, 0.0])
                
                # Outer 5 colors
                outside_features = []
                for cid in range(5, 10):
                    match_pixels = (c_grid == cid) & out_mask
                    if np.any(match_pixels):
                        mean_col = np.mean(batch_grids[b][match_pixels], axis=0)
                        outside_features.append([mean_col[0], mean_col[1], mean_col[2], 1.0])
                    else:
                        outside_features.append([0.0, 0.0, 0.0, 0.0])
                
                # Sort by luminance (R+G+B) for deterministic ordering
                inside_arr = np.array(inside_features)
                outside_arr = np.array(outside_features)
                
                in_sorted = inside_arr[np.argsort(inside_arr[:, :3].sum(axis=1))]
                out_sorted = outside_arr[np.argsort(outside_arr[:, :3].sum(axis=1))]
                
                combined = np.concatenate([in_sorted, out_sorted], axis=0)  # (10, 4)
                targets_list.append(combined.flatten())

            targets_b = np.array(targets_list, dtype=np.float32)

            for b in range(current_batch_size):
                X_images.append(batch_grids_transposed[b])
                y_colors.append(targets_b[b])

        gen_elapsed = time.time() - start_t
        print(f"✅ Data generation completed in {gen_elapsed:.2f}s ({num_samples:,} samples)")

        X_arr = np.array(X_images, dtype=np.float32)
        y_arr = np.array(y_colors, dtype=np.float32)

        self._visualize_training_samples_grid(X_arr, y_arr, num_display=10)
        return X_arr, y_arr

    def train_model(self, epochs=40, batch_size=256, num_samples=50000, patience=8):
        self._log(f"Training RegionalColorExtractor v3 with {num_samples:,} samples...")
        
        X_imgs, y_cols = self._generate_training_data(num_samples)

        dataset = TensorDataset(torch.tensor(X_imgs), torch.tensor(y_cols))
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self.to(self.device)
        
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        criterion_rgb = nn.MSELoss(reduction='none')
        criterion_flag = nn.BCEWithLogitsLoss(reduction='none')

        best_loss = float('inf')
        patience_counter = 0

        print(f"\n--- 🚀 RegionalColorExtractor v3 Training Started (Separated RGB & Flag Losses) ---")
        total_train_start = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            self.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
            for b_x, b_y in train_pbar:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                
                preds = self(b_x)  # (B, 10, 4) -> RGB [0,1], Flag Logits
                targets = b_y.view(-1, 10, 4)
                
                preds_rgb = preds[:, :, :3]
                preds_flag_logits = preds[:, :, 3]
                
                targets_rgb = targets[:, :, :3]
                targets_flag = targets[:, :, 3]  # 0.0 or 1.0
                
                # Flag Loss (BCEWithLogitsLoss)
                loss_flag = criterion_flag(preds_flag_logits, targets_flag).mean()
                
                # RGB Loss strictly masked to colors that actually exist (targets_flag > 0.5)
                flag_mask = targets_flag.unsqueeze(-1)  # (B, 10, 1)
                rgb_sq_diff = ((preds_rgb - targets_rgb) ** 2) * flag_mask
                loss_rgb = rgb_sq_diff.sum() / (flag_mask.sum() * 3.0 + 1e-6)
                
                loss = loss_rgb + 0.5 * loss_flag
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * b_x.size(0)
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            avg_train_loss = train_loss / len(train_dataset)

            self.eval()
            val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val  ", leave=False)
            with torch.no_grad():
                for b_x, b_y in val_pbar:
                    b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                    preds = self(b_x)
                    targets = b_y.view(-1, 10, 4)
                    
                    preds_rgb = preds[:, :, :3]
                    preds_flag_logits = preds[:, :, 3]
                    targets_rgb = targets[:, :, :3]
                    targets_flag = targets[:, :, 3]
                    
                    loss_flag = criterion_flag(preds_flag_logits, targets_flag).mean()
                    flag_mask = targets_flag.unsqueeze(-1)
                    rgb_sq_diff = ((preds_rgb - targets_rgb) ** 2) * flag_mask
                    loss_rgb = rgb_sq_diff.sum() / (flag_mask.sum() * 3.0 + 1e-6)
                    
                    loss = loss_rgb + 0.5 * loss_flag
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

    def extract_colors_from_pil(self, pil_img, device=None):
        """
        At inference time: Extracts assuming luminance sorting and returns only those with valid existence flags as top entries.
        """
        if device is None:
            device = self.device
            
        self.eval()
        img_resized = pil_img.resize((self.target_size, self.target_size))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        
        tensor_img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = self.forward(tensor_img).cpu().squeeze(0).numpy()  # (10, 4)
            
        inside_preds = pred[:5]   # Object side
        outside_preds = pred[5:]  # Background side
        
        # Extract ones with existence flag logit > 0 (or probability > 0.5 after sigmoid)
        inside_colors = [c[:3] for c in inside_preds if c[3] > 0.0]
        if not inside_colors:
            inside_colors = [c[:3] for c in inside_preds[:3]]
            
        outside_colors = [c[:3] for c in outside_preds if c[3] > 0.0]
        if not outside_colors:
            outside_colors = [c[:3] for c in outside_preds[:3]]
        
        return {
            "group_1_inside_circle": {
                "description": "Colors inside the center circle (with existence filter)",
                "colors": inside_colors[:5]
            },
            "group_2_outside_circle": {
                "description": "Colors outside the center circle",
                "colors": outside_colors[:5]
            },
            "all_colors": [c[:3] for c in pred]
        }