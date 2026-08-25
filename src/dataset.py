import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json

class AmodalDataset(Dataset):
    def __init__(self, data_dir, mode="toy", image_size=(256, 256), coco_img_dir=None, augment=None):
        """
        
        Args:
            data_dir (str): Đường dẫn đến file .json nhãn COCOA tương ứng (Train/Val/Test).
            mode (str): Chế độ chạy ("toy", "pix2gestalt", "cocoa_train", "cocoa_val", "cocoa_test").
            image_size (tuple): Kích thước chuẩn hóa đầu vào (H, W).
            coco_img_dir (str): Đường dẫn đến thư mục chứa TOÀN BỘ ảnh phẳng (Format: COCO_xxxx2014_xxxx.jpg).
            augment (bool | None): Bật/tắt augmentation (hiện tại: random horizontal flip đồng bộ
                trên I/M_v/M_a). None (mặc định) = tự động bật cho mode train ("pix2gestalt",
                "cocoa_train"), tắt cho val/test/toy -- để chống overfitting trên các tập train nhỏ
                (đặc biệt COCOA) mà không làm rò rỉ augmentation vào lúc đánh giá.
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size 
        self.coco_img_dir = coco_img_dir
        self.samples = [] 

        if augment is None:
            self.augment = mode in ("pix2gestalt", "cocoa_train")
        else:
            self.augment = augment
        
        # ==========================================
        # CHẾ ĐỘ 1: TOY DATASET (Sinh dữ liệu giả lập)
        # ==========================================
        if self.mode == "toy":
            print("🚀 Đang khởi tạo Toy Dataset (Dữ liệu giả lập hình khối)...")
            self.num_samples = 100 
            
        # ==========================================
        # CHẾ ĐỘ 2: PIX2GESTALT 
        # ==========================================
        elif self.mode == "pix2gestalt":
            print("⏳ Đang quét và đối chiếu chéo dữ liệu Pix2Gestalt ...")
            self.occlusion_dir = os.path.join(data_dir, 'occlusion')
            self.visible_mask_dir = os.path.join(data_dir, 'visible_object_mask')
            self.whole_mask_dir = os.path.join(data_dir, 'whole_mask')
            
            occ_ids = {f.split('_')[0] for f in os.listdir(self.occlusion_dir) if f.endswith('.png')}
            vis_ids = {f.split('_')[0] for f in os.listdir(self.visible_mask_dir) if f.endswith('.png')}
            whole_ids = {f.split('_')[0] for f in os.listdir(self.whole_mask_dir) if f.endswith('.png')}
            
            valid_ids = occ_ids & vis_ids & whole_ids
            self.base_ids = sorted(list(valid_ids))
            self.num_samples = len(self.base_ids)
            print(f"✅ Đã tìm thấy {self.num_samples} mẫu hợp lệ.")

        # ==========================================
        # CHẾ ĐỘ 3: COCOA (Thư mục phẳng giữ nguyên tên gốc)
        # ==========================================
        elif self.mode.startswith("cocoa"):
            print(f"⏳ Đang phân tích đồ thị JSON của {self.mode.upper()} ...")
            if not self.coco_img_dir:
                raise ValueError("LỖI: Chế độ COCOA yêu cầu phải truyền biến 'coco_img_dir'.")
            
            # Đọc danh sách tất cả các file ảnh thực tế đang có trong thư mục phẳng trên RAM
            print("💾 Đang lập chỉ mục thư mục ảnh phẳng...")
            self.available_images = set(os.listdir(self.coco_img_dir))
            
            with open(self.data_dir, 'r') as f:
                cocoa_data = json.load(f)
            
            skipped_images_count = 0
            skipped_objects_count = 0
            
            # Quét từng bức ảnh trong file JSON
            for ann in cocoa_data['annotations']:
                # Lấy trực tiếp tên file từ URL (Ví dụ: "COCO_test2014_000000054456.jpg")
                img_filename = ann['url'].split('/')[-1] 
                
                # CHUYỂN ĐỔI CHẾ ĐỘ THỬ NGHIỆM: Kiểm tra trực tiếp tên file gốc trong tập hợp RAM
                if img_filename not in self.available_images:
                    # Nếu thiếu ảnh trên đĩa, tự động bỏ qua toàn bộ vật thể thuộc ảnh này
                    skipped_images_count += 1
                    skipped_objects_count += len([r for r in ann['regions'] if r.get('isStuff', 0) != 1])
                    continue
                
                # Giải mã đồ thị che khuất (Depth Constraints)
                depth_constraints = ann.get('depth_constraint', '')
                occluders_of = {} 
                
                if depth_constraints:
                    pairs = depth_constraints.split(',')
                    for pair in pairs:
                        if '-' in pair:
                            front, back = pair.split('-')
                            front, back = int(front), int(back)
                            if back not in occluders_of:
                                occluders_of[back] = []
                            occluders_of[back].append(front)

                # Duyệt qua các vùng vật thể (Regions)
                regions = ann['regions']
                for region in regions:
                    if region.get('isStuff', 0) == 1: 
                        continue
                        
                    region_order = region.get('order')
                    if region_order is not None:
                        self.samples.append({
                            'img_filename': img_filename,
                            'region': region,
                            'region_order': region_order,
                            'all_regions': regions,
                            'occluders': occluders_of.get(region_order, [])
                        })
                    
            self.num_samples = len(self.samples)
            print(f"✅ Đã giải mã thành công {self.num_samples} vật thể từ COCOA.")
            if skipped_images_count > 0:
                print(f"⚠️ [RESEARCH MODE] Đã tự động phát hiện và bỏ qua {skipped_images_count} ảnh bị thiếu trên ổ đĩa "
                      f"(Tương ứng với {skipped_objects_count} vật thể bị loại bỏ khỏi danh sách huấn luyện/đánh giá).")
        else:
            raise ValueError(f"Chế độ mode='{self.mode}' không được hỗ trợ!")

    def __len__(self):
        return self.num_samples

    def _poly_to_mask(self, polygons, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        if isinstance(polygons[0], list):
            pts = [np.array(p, np.int32).reshape((-1, 2)) for p in polygons]
        else:
            pts = [np.array(polygons, np.int32).reshape((-1, 2))]
        cv2.fillPoly(mask, pts, 1)
        return mask

    def _maybe_hflip(self, img, m_v, m_a):
        """Random horizontal flip đồng bộ, xác suất 50%. Xem docstring _augment_all()."""
        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, ...])
            m_v = np.ascontiguousarray(m_v[:, ::-1])
            m_a = np.ascontiguousarray(m_a[:, ::-1])
        return img, m_v, m_a

    def _maybe_scale_crop(self, img, m_v, m_a):
        """
        Random crop một vùng con (85-100% diện tích gốc) rồi resize lại đúng self.image_size,
        ĐỒNG BỘ trên cả 3 (dùng chung 1 crop box, img nội suy linear, mask nội suy nearest
        để giữ mask nhị phân sạch, không sinh giá trị trung gian 0<x<1).
        """
        if not self.augment:
            return img, m_v, m_a

        H, W = img.shape[:2]
        scale = np.random.uniform(0.85, 1.0)
        new_h, new_w = int(round(H * scale)), int(round(W * scale))
        if new_h >= H or new_w >= W or new_h <= 0 or new_w <= 0:
            return img, m_v, m_a

        top = np.random.randint(0, H - new_h + 1)
        left = np.random.randint(0, W - new_w + 1)

        img_c = img[top:top + new_h, left:left + new_w]
        m_v_c = m_v[top:top + new_h, left:left + new_w]
        m_a_c = m_a[top:top + new_h, left:left + new_w]

        img = cv2.resize(img_c, (W, H), interpolation=cv2.INTER_LINEAR)
        m_v = cv2.resize(m_v_c, (W, H), interpolation=cv2.INTER_NEAREST)
        m_a = cv2.resize(m_a_c, (W, H), interpolation=cv2.INTER_NEAREST)
        return img, m_v, m_a

    def _maybe_color_jitter(self, img):
        """
        Jitter brightness/contrast nhẹ (±15%), CHỈ áp dụng lên ảnh RGB -- không đụng vào
        mask (occlusion không phụ thuộc màu sắc/độ sáng). img: (H,W,3), giá trị 0-255.
        """
        if not self.augment:
            return img
        img = img.astype(np.float32)
        brightness = np.random.uniform(0.85, 1.15)
        contrast = np.random.uniform(0.85, 1.15)
        mean_val = img.mean()
        img = (img - mean_val) * contrast + mean_val
        img = img * brightness
        return np.clip(img, 0, 255)

    def _maybe_rotate(self, img, m_v, m_a, max_angle=15):
        """
        Xoay ngẫu nhiên góc nhỏ (±max_angle độ) quanh tâm ảnh, ĐỒNG BỘ trên cả 3.
        Dùng cv2.warpAffine, img nội suy linear (BORDER_REFLECT tránh viền đen nhân tạo),
        mask nội suy nearest + BORDER_CONSTANT=0 (vùng trống ngoài mask sau khi xoay coi
        như không thuộc mask, hợp lý về mặt occlusion).
        """
        if not self.augment or np.random.rand() < 0.5:
            return img, m_v, m_a

        H, W = img.shape[:2]
        angle = np.random.uniform(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)

        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m_v = cv2.warpAffine(m_v, M, (W, H), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m_a = cv2.warpAffine(m_a, M, (W, H), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, m_v, m_a

    def _maybe_synthetic_occlusion(self, m_v, m_a, prob=0.4, max_patches=2, patch_scale_range=(0.05, 0.25)):
        """
        Augmentation ĐẶC THÙ cho amodal segmentation (không có ở generic augmentation
        pipelines): che thêm ngẫu nhiên 1-2 mảng hình chữ nhật TRONG vùng M_v (visible),
        đặt về 0 -- trong khi M_a (ground truth amodal) GIỮ NGUYÊN KHÔNG ĐỔI.

        Mục đích: buộc model, với đúng cùng ground truth M_a, phải học tái tạo từ NHIỀU
        kiểu che khuất M_v khác nhau (nhân bản đa dạng occlusion pattern mà không cần
        thêm ảnh mới) -- nhắm trực tiếp vào khả năng suy luận vùng bị che (mIoU_inv),
        đúng nút thắt hiện tại thay vì chỉ tăng đa dạng ảnh/màu sắc chung chung.

        CHỈ sửa m_v, không đụng đến m_a hay img (ảnh RGB vẫn hiển thị vật thể đầy đủ,
        chỉ có "nhãn visible" bị che thêm -- hợp lý vì input model là (I, M_v), không
        phải I đã bị crop vật lý).
        """
        if not self.augment or np.random.rand() > prob:
            return m_v

        H, W = m_v.shape
        ys, xs = np.where(m_v > 0)
        if len(ys) == 0:
            return m_v

        m_v = m_v.copy()
        n_patches = np.random.randint(1, max_patches + 1)
        for _ in range(n_patches):
            idx = np.random.randint(len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])
            patch_h = max(1, int(H * np.random.uniform(*patch_scale_range)))
            patch_w = max(1, int(W * np.random.uniform(*patch_scale_range)))
            y0, y1 = max(0, cy - patch_h // 2), min(H, cy + patch_h // 2)
            x0, x1 = max(0, cx - patch_w // 2), min(W, cx + patch_w // 2)
            m_v[y0:y1, x0:x1] = 0
        return m_v

    def _augment_all(self, img, m_v, m_a):
        """
        Gộp toàn bộ augmentation, gọi 1 lần duy nhất ngay sau resize, trước khi normalize.
        Thứ tự: hflip -> scale/crop -> rotate (ảnh hưởng không gian, áp cho cả 3)
        -> color jitter (chỉ ảnh) -> synthetic occlusion (chỉ M_v, KHÔNG đụng M_a).
        Tự động no-op nếu self.augment=False (val/test).
        """
        img, m_v, m_a = self._maybe_hflip(img, m_v, m_a)
        img, m_v, m_a = self._maybe_scale_crop(img, m_v, m_a)
        img, m_v, m_a = self._maybe_rotate(img, m_v, m_a)
        img = self._maybe_color_jitter(img)
        m_v = self._maybe_synthetic_occlusion(m_v, m_a)
        return img, m_v, m_a

    def __getitem__(self, idx):
        if self.mode == "toy":
            return self._generate_toy_data()
            
        elif self.mode == "pix2gestalt":
            base_id = self.base_ids[idx]
            img_path = os.path.join(self.occlusion_dir, f"{base_id}_occlusion.png")
            m_v_path = os.path.join(self.visible_mask_dir, f"{base_id}_visible_mask.png")
            m_a_path = os.path.join(self.whole_mask_dir, f"{base_id}_whole_mask.png")
            
            img = cv2.imread(img_path)
            m_v = cv2.imread(m_v_path, cv2.IMREAD_GRAYSCALE)
            m_a = cv2.imread(m_a_path, cv2.IMREAD_GRAYSCALE)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            m_v = cv2.resize(m_v, self.image_size, interpolation=cv2.INTER_NEAREST)
            m_a = cv2.resize(m_a, self.image_size, interpolation=cv2.INTER_NEAREST)

            img, m_v, m_a = self._augment_all(img, m_v, m_a)

            img = img.astype(np.float32) / 255.0
            m_v = (m_v > 127).astype(np.float32)
            m_a = (m_a > 127).astype(np.float32)
            
            img = np.transpose(img, (2, 0, 1))
            m_v = np.expand_dims(m_v, axis=0)
            m_a = np.expand_dims(m_a, axis=0)
            return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a)
            
        elif self.mode.startswith("cocoa"):
            sample = self.samples[idx]
            img_filename = sample['img_filename'] 
            
            # ĐỌC TRỰC TIẾP TỪ THƯ MỤC PHẲNG - KHÔNG CHIA NHÁNH TRAIN/VAL/TEST
            img_path = os.path.join(self.coco_img_dir, img_filename)
                
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img.shape[:2]
            
            # Dựng Amodal Mask
            target_poly = sample['region']['segmentation']
            amodal_mask = self._poly_to_mask(target_poly, h_orig, w_orig)
            
            # Dựng Occluder Mask
            occluder_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            all_regions = sample['all_regions']
            
            for occ_order in sample['occluders']:
                for r in all_regions:
                    if r.get('order') == occ_order:
                        occ_poly = r['segmentation']
                        occ_mask = self._poly_to_mask(occ_poly, h_orig, w_orig)
                        occluder_mask = np.logical_or(occluder_mask, occ_mask).astype(np.uint8)
                        break 
                        
            # Tính toán Modal Mask
            modal_mask = amodal_mask.copy()
            modal_mask[occluder_mask == 1] = 0
            
            # Tiền xử lý & Trả về Tensor chuẩn PyTorch Layout (C, H, W)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            m_v = cv2.resize(modal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            m_a = cv2.resize(amodal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            img, m_v, m_a = self._augment_all(img, m_v, m_a)

            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            m_v = m_v.astype(np.float32)
            m_a = m_a.astype(np.float32)
            
            img = np.transpose(img, (2, 0, 1))
            m_v = np.expand_dims(m_v, axis=0)
            m_a = np.expand_dims(m_a, axis=0)
            
            return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a)