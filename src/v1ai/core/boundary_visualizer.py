import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

class BoundaryVisualizer:
    def __init__(self, config, patch_size=9, max_image_size=1028):
        self.config = config
        self.device = getattr(config, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"🚀 [BoundaryVisualizer] Using device: {self.device}")
        
        self.patch_size = getattr(config, "patch_size", patch_size)
        self.max_border_colors = 10
        self.max_object_colors = 5
        self.max_bg_colors = 5
        self.max_image_size = max_image_size
        
        self.retention_percent = float(getattr(
            getattr(config, "app", None), 
            "boundary_retention_percent", 
            getattr(config, "boundary_retention_percent", 20.0)
        ))
        
        self.space_dir = getattr(config, "space_dir", "./")
        self.model_dir = os.path.join(self.space_dir, "models")
        
        suffix = f"_{self.patch_size}x{self.patch_size}" if self.patch_size != 9 else ""
        self.boundary_model_path = os.path.join(self.model_dir, f"boundary_confidence_classifier{suffix}.pth")
        self.extractor_model_path = os.path.join(self.model_dir, f"object_extractor_filter{suffix}.pth")
        self.color_model_path = os.path.join(self.model_dir, "regional_color_extractor.pth")
        
    def _load_models(self):
        print("📂 Loading models onto device...")
        t_load_start = time.time()
        if not os.path.exists(self.boundary_model_path):
            raise FileNotFoundError(f"⚠️ BoundaryFilter model not found at [{self.boundary_model_path}].")
        if not os.path.exists(self.extractor_model_path):
            raise FileNotFoundError(f"⚠️ ObjectExtractorFilter model not found at [{self.extractor_model_path}].")
        if not os.path.exists(self.color_model_path):
            raise FileNotFoundError(f"⚠️ RegionalColorExtractor model not found at [{self.color_model_path}].")
        
        from .boundary_filter import BoundaryFilter
        from .object_extractor_filter import ObjectExtractorFilter
        from .regional_color_extractor import RegionalColorExtractor

        boundary_model = BoundaryFilter(patch_size=self.patch_size, max_query_colors=self.max_border_colors).to(self.device)
        try:
            boundary_model.load_state_dict(torch.load(self.boundary_model_path, map_location=self.device, weights_only=True))
        except TypeError:
            boundary_model.load_state_dict(torch.load(self.boundary_model_path, map_location=self.device))
        boundary_model.eval()

        extractor_model = ObjectExtractorFilter(
            patch_size=self.patch_size, 
            max_object_colors=self.max_object_colors, 
            max_bg_colors=self.max_bg_colors
        ).to(self.device)
        try:
            extractor_model.load_state_dict(torch.load(self.extractor_model_path, map_location=self.device, weights_only=True))
        except TypeError:
            extractor_model.load_state_dict(torch.load(self.extractor_model_path, map_location=self.device))
        extractor_model.eval()
        
        color_model = RegionalColorExtractor(target_size=32).to(self.device)
        try:
            color_model.load_state_dict(torch.load(self.color_model_path, map_location=self.device, weights_only=True))
        except TypeError:
            color_model.load_state_dict(torch.load(self.color_model_path, map_location=self.device))
        color_model.eval()

        print(f"✅ Models loaded successfully in {time.time() - t_load_start:.2f}s")
        return boundary_model, extractor_model, color_model

    def _resize_if_needed(self, pil_img):
        w, h = pil_img.size
        if max(w, h) > self.max_image_size:
            if w > h:
                new_w = self.max_image_size
                new_h = int(h * (self.max_image_size / w))
            else:
                new_h = self.max_image_size
                new_w = int(w * (self.max_image_size / h))
            print(f"🖼️ Resizing image from ({w}x{h}) to ({new_w}x{new_h}) for performance.")
            return pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return pil_img

    def _extract_crop_by_mode(self, pil_img, mode):
        W, H = pil_img.size
        cw = int(W * 0.3)
        ch = int(H * 0.3)
        
        if mode == 'full':
            return pil_img
        elif mode == 'top-right':
            return pil_img.crop((W - cw, 0, W, ch))
        elif mode == 'top-left':
            return pil_img.crop((0, 0, cw, ch))
        elif mode == 'bottom-right':
            return pil_img.crop((W - cw, H - ch, W, H))
        elif mode == 'bottom-left':
            return pil_img.crop((0, H - ch, cw, H))
        return pil_img

    def _get_colors(self, pil_img, color_model, mode):
        crop_img = self._extract_crop_by_mode(pil_img, mode)
        res = color_model.extract_colors_from_pil(crop_img, device=self.device)
        obj_colors = res.get("group_1_inside_circle", {}).get("colors", [])[:5]
        bg_colors = res.get("group_2_outside_circle", {}).get("colors", [])[:5]
        return obj_colors, bg_colors

    def _infer_boundary_map(self, pil_img, boundary_model, border_colors_list, desc="Boundary"):
        W, H = pil_img.size
        tensor_img = torch.tensor(np.array(pil_img, dtype=np.float32) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        ps = self.patch_size
        pad = ps // 2
        padded_img = F.pad(tensor_img, (pad, pad, pad, pad), mode='replicate')
        
        patches_raw = F.unfold(padded_img, kernel_size=ps, stride=1)
        L = patches_raw.shape[2]
        patches_tensor = patches_raw.view(1, 3, ps, ps, L).permute(0, 4, 1, 2, 3).squeeze(0)

        query_vector = np.zeros(self.max_border_colors * 4, dtype=np.float32)
        limited_colors = border_colors_list[:self.max_border_colors]
        for idx, c in enumerate(limited_colors):
            query_vector[idx * 4 : idx * 4 + 3] = c
            query_vector[idx * 4 + 3] = 1.0

        q_tensor = torch.tensor(query_vector, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(L, 1)

        batch_size = 4096
        all_edge_probs = []
        all_rgb_preds = []
        
        with torch.no_grad():
            for i in range(0, patches_tensor.shape[0], batch_size):
                b_patches = patches_tensor[i:i+batch_size].to(self.device)
                b_queries = q_tensor[i:i+batch_size].to(self.device)
                
                preds = boundary_model(b_patches, b_queries)
                edge_prob = preds[:, 0:1]
                rgb_pred = preds[:, 1:]
                
                all_edge_probs.append(edge_prob)
                all_rgb_preds.append(rgb_pred)
        
        edge_tensor = torch.cat(all_edge_probs, dim=0)
        rgb_tensor = torch.cat(all_rgb_preds, dim=0)
        
        scores_flat = edge_tensor.detach().cpu().squeeze(1)
        retention_ratio = self.retention_percent / 100.0
        quantile_val = max(0.0, min(1.0, 1.0 - retention_ratio))
        dynamic_threshold = torch.quantile(scores_flat, quantile_val).item()
        dynamic_threshold = max(dynamic_threshold, 0.40)
        
        confidence_map = edge_tensor.view(H, W).detach().cpu().numpy()
        color_map = rgb_tensor.view(H, W, 3).detach().cpu().numpy()
        
        return confidence_map, color_map, dynamic_threshold

    def _infer_object_extractor_map(self, pil_img, boundary_model, extractor_model, obj_colors_list, bg_colors_list):
        W, H = pil_img.size
        
        border_colors_full = obj_colors_list + bg_colors_list
        confidence_map, color_map, dyn_thresh = self._infer_boundary_map(pil_img, boundary_model, border_colors_full, desc="Obj-Boundary")
        edge_mask = (confidence_map >= dyn_thresh)

        tensor_img = torch.tensor(np.array(pil_img, dtype=np.float32) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        ps = self.patch_size
        pad = ps // 2
        padded_img = F.pad(tensor_img, (pad, pad, pad, pad), mode='replicate')

        patches_raw = F.unfold(padded_img, kernel_size=ps, stride=1)
        L = patches_raw.shape[2]
        patches_tensor = patches_raw.view(1, 3, ps, ps, L).permute(0, 4, 1, 2, 3).squeeze(0)

        obj_vector = np.zeros(self.max_object_colors * 4, dtype=np.float32)
        limited_obj = obj_colors_list[:self.max_object_colors]
        for idx, c in enumerate(limited_obj):
            obj_vector[idx*4 : idx*4+3] = c
            obj_vector[idx*4+3] = 1.0
        obj_q_tensor = torch.tensor(obj_vector, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(L, 1)

        bg_vector = np.zeros(self.max_bg_colors * 4, dtype=np.float32)
        limited_bg = bg_colors_list[:self.max_bg_colors]
        for idx, c in enumerate(limited_bg):
            bg_vector[idx*4 : idx*4+3] = c
            bg_vector[idx*4+3] = 1.0
        bg_q_tensor = torch.tensor(bg_vector, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(L, 1)

        batch_size = 4096
        all_is_bg_logits = []
        
        extractor_model.eval()
        with torch.no_grad():
            for i in range(0, patches_tensor.shape[0], batch_size):
                b_patches = patches_tensor[i:i+batch_size].to(self.device)
                b_obj_q = obj_q_tensor[i:i+batch_size].to(self.device)
                b_bg_q = bg_q_tensor[i:i+batch_size].to(self.device)
                
                center_idx = ps // 2
                center_colors = b_patches[:, :, center_idx, center_idx]
                center_expanded = center_colors.unsqueeze(1)
                
                obj_rgb_list = [b_obj_q[:, idx*4:idx*4+3] for idx in range(5)]
                bg_rgb_list = [b_bg_q[:, idx*4:idx*4+3] for idx in range(5)]
                
                obj_palettes = torch.stack(obj_rgb_list, dim=1)
                bg_palettes = torch.stack(bg_rgb_list, dim=1)
                
                obj_dists = torch.norm(center_expanded - obj_palettes, dim=-1)
                bg_dists = torch.norm(center_expanded - bg_palettes, dim=-1)
                
                obj_min_dist = torch.min(obj_dists, dim=-1, keepdim=True)[0]
                obj_mean_dist = torch.mean(obj_dists, dim=-1, keepdim=True)
                bg_min_dist = torch.min(bg_dists, dim=-1, keepdim=True)[0]
                bg_mean_dist = torch.mean(bg_dists, dim=-1, keepdim=True)
                
                b_dist = torch.cat([obj_min_dist, obj_mean_dist, bg_min_dist, bg_mean_dist], dim=1)
                
                preds = extractor_model(b_patches, b_obj_q, b_bg_q, b_dist)
                is_bg_logit = preds[:, 0:1]
                all_is_bg_logits.append(is_bg_logit)
                
        is_bg_tensor = torch.cat(all_is_bg_logits, dim=0)
        bg_probs = torch.sigmoid(is_bg_tensor).view(H, W).detach().cpu().numpy()
        
        is_background = (bg_probs > 0.5) & (~edge_mask)
        object_mask = ~is_background
        return object_mask

    def _compute_common_boundary_master_image(self, img_np, all_boundary_display_maps):
        base_bg_color = np.array([0.96, 0.98, 0.99])
        intersect_mask = np.ones(img_np.shape[:2], dtype=bool)
        
        for disp_map in all_boundary_display_maps:
            diff = np.sqrt(np.sum((disp_map - base_bg_color)**2, axis=2))
            is_edge = diff > 0.04 
            intersect_mask = intersect_mask & is_edge
            
        common_boundary_img = np.ones_like(img_np) * 1.0  
        kernel = np.ones((3, 3), np.uint8)
        thick_intersect_mask = cv2.dilate(intersect_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        common_boundary_img[thick_intersect_mask] = img_np[thick_intersect_mask] * 0.2  
        return common_boundary_img

    def visualize_sample_images(self):
        sample_dir = os.path.join(self.config.dataset_dir, "sample_image")
        if not os.path.exists(sample_dir):
            print(f"⚠️ sample_image directory not found: [{sample_dir}]")
            return

        image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("⚠️ No image files found.")
            return

        try:
            boundary_model, extractor_model, color_model = self._load_models()
        except Exception as e:
            print(e)
            return

        output_dir = os.path.join(self.space_dir, "boundary_filtered_results")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n--- 🚀 4-Row 5-Col Pipeline Visualization Started ({len(image_files)} images found) ---")
        
        for filename in image_files:
            print(f"\n📸 Processing image: [{filename}]")
            t_img_start = time.time()
            
            img_path = os.path.join(sample_dir, filename)
            raw_pil_img = Image.open(img_path).convert("RGB")
            pil_img = self._resize_if_needed(raw_pil_img)
            img_np = np.array(pil_img, dtype=np.float32) / 255.0

            col_modes = ['full', 'top-right', 'top-left', 'bottom-right', 'bottom-left']
            col_descriptions = [
                "[Col 1: Full Screen]\n- Extract 5 object & 5 BG colors from full image.",
                "[Col 2: Top-Right Corner]\n- Extract 5 object & 5 BG colors from top-right corner.",
                "[Col 3: Top-Left Corner]\n- Extract 5 object & 5 BG colors from top-left corner.",
                "[Col 4: Bottom-Right Corner]\n- Extract 5 object & 5 BG colors from bottom-right corner.",
                "[Col 5: Bottom-Left Corner]\n- Extract 5 object & 5 BG colors from bottom-left corner."
            ]

            fig, axes = plt.subplots(4, 5, figsize=(25, 20))

            axes[0, 0].imshow(pil_img)
            axes[0, 0].set_title("Original Image\n(Full View)", fontsize=10, fontweight='bold')
            axes[0, 0].axis('off')

            for col_idx in range(1, 5):
                ax_empty = axes[0, col_idx]
                ax_empty.axis('off')
                ax_empty.text(0.5, 0.5, col_descriptions[col_idx], transform=ax_empty.transAxes,
                            ha='center', va='center', fontsize=7.5, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CED4DA'))

            all_boundary_display_maps = []

            for col_idx, mode in enumerate(col_modes):
                t_mode_start = time.time()
                print(f"  👉 Processing Column {col_idx+1}/5 [Mode: {mode}]...")

                obj_cols, bg_cols = self._get_colors(pil_img, color_model, mode=mode)
                border_cols_full = obj_cols + bg_cols

                # 1. Boundary Map Inference
                conf_map, col_map, dyn_thresh = self._infer_boundary_map(pil_img, boundary_model, border_cols_full)
                
                # 背景色をより薄く設定 (非常に明るい薄い青/白み)
                base_bg_color = np.array([0.96, 0.98, 0.99], dtype=np.float32)
                display_map = np.ones((*conf_map.shape, 3), dtype=np.float32) * base_bg_color[None, None, :]

                # 閾値以上のエッジマスクを作成し、膨張（dilate）で線を太くする
                edge_mask = (conf_map >= dyn_thresh).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                thick_edge_mask = cv2.dilate(edge_mask, kernel, iterations=1) > 0

                # スコアによる輝度の変化を入れつつ、明るめでくっきりした（濃い）表示に調整
                score_adjusted = np.clip(conf_map * 1.2 + 0.2, 0.0, 1.0)

                # ベクトル演算による高速一括描画
                for c in range(3):
                    channel_bg = base_bg_color[c]
                    channel_fg = col_map[:, :, c]
                    val = channel_fg * score_adjusted + channel_bg * (1.0 - score_adjusted)
                    display_map[:, :, c] = np.where(thick_edge_mask, val, channel_bg)

                display_map = np.clip(display_map, 0.0, 1.0)
                all_boundary_display_maps.append(display_map)

                ax_b = axes[1, col_idx]
                ax_b.imshow(display_map)
                ax_b.set_title(f"Boundary Map ({mode})", fontsize=8, fontweight='bold')
                ax_b.axis('off')

                b_hex_str = " ".join([f"■#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for c in obj_cols[:3]])
                ax_b.text(0.5, -0.12, f"Obj Colors:\n{b_hex_str}", transform=ax_b.transAxes,
                          ha='center', va='top', fontsize=6, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

                # 2. Object Extractor Inference
                object_mask = self._infer_object_extractor_map(
                    pil_img, boundary_model, extractor_model, obj_cols, bg_cols
                )

                extracted_img = img_np.copy()
                extracted_img[~object_mask] = np.array([1.0, 1.0, 1.0])

                ax_e = axes[2, col_idx]
                ax_e.imshow(extracted_img)
                ax_e.set_title(f"Object Extracted", fontsize=8, fontweight='bold', color='darkblue')
                ax_e.axis('off')

                bg_hex_str = " ".join([f"■#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for c in bg_cols[:3]])
                ax_e.text(0.5, -0.12, f"BG Colors:\n{bg_hex_str}", transform=ax_e.transAxes,
                          ha='center', va='top', fontsize=6, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
                
                print(f"     ✅ Mode [{mode}] finished in {time.time() - t_mode_start:.2f}s")

            common_boundary_master = self._compute_common_boundary_master_image(img_np, all_boundary_display_maps)

            ax_m = axes[3, 0]
            ax_m.imshow(common_boundary_master)
            ax_m.set_title("Common Boundary Master\n(5-Map Intersection)", fontsize=9, fontweight='bold', color='darkred')
            ax_m.axis('off')
            ax_m.text(0.5, -0.12, "Intersection of All 5 Boundary Maps (Enhanced)", transform=ax_m.transAxes,
                      ha='center', va='top', fontsize=6, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

            for col_idx in range(1, 5):
                ax_sub = axes[3, col_idx]
                ax_sub.axis('off')
                ax_sub.text(0.5, 0.5, "<- Extracted only the boundaries\n common to all 5 maps in Row 2", transform=ax_sub.transAxes,
                            ha='center', va='center', fontsize=8, color='gray', fontweight='bold')

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"pipeline_4row_5col_grid_{filename}")
            plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  🎉 Finished [{filename}] in {time.time() - t_img_start:.2f}s. Saved to {out_path}")

        print(f"\n📂 All results saved to: [{output_dir}]\n")