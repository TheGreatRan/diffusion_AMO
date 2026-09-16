import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json


class AmodalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mode="toy",
        image_size=(256, 256),
        coco_img_dir=None,
        augment=None,
        # ------------------------------------------------------------
        # NEW: control riêng từng loại augmentation.
        # None = kế thừa theo `augment` (hành vi cũ). True/False = ép cứng
        # loại đó bật/tắt bất kể `augment`.
        # ------------------------------------------------------------
        augment_hflip=None,
        augment_scale_crop=None,
        augment_rotate=None,
        augment_color_jitter=None,
        augment_synthetic_occlusion=None,
        # ------------------------------------------------------------
        # NEW: lọc sample lỗi/degenerate ở COCOA (chỉ ảnh hưởng mode "cocoa*")
        # ------------------------------------------------------------
        filter_degenerate=True,       # luôn nên bật: loại annotation vỡ (area~0)
        filter_fully_occluded=False,  # mặc định TẮT: vật thể che 100% vẫn là sample hợp lệ
        min_area_px=4,                # ngưỡng diện tích tối thiểu để coi là "không rỗng"
    ):
        """
        Args:
            data_dir (str): Đường dẫn đến file .json nhãn COCOA tương ứng (Train/Val/Test).
            mode (str): Chế độ chạy ("toy", "pix2gestalt", "cocoa_train", "cocoa_val", "cocoa_test").
            image_size (tuple): Kích thước chuẩn hóa đầu vào (H, W).
            coco_img_dir (str): Đường dẫn đến thư mục chứa TOÀN BỘ ảnh phẳng.
            augment (bool | None): Master switch. None (mặc định) = tự bật cho mode train
                ("pix2gestalt", "cocoa_train"), tắt cho val/test/toy.
            augment_hflip / augment_scale_crop / augment_rotate / augment_color_jitter /
                augment_synthetic_occlusion (bool | None): Override riêng lẻ từng loại,
                bất kể `augment`. Dùng để dựng preset, ví dụ:
                  - Paper-faithful (không augmentation gì cả):
                        augment=False
                  - Paper-faithful + chỉ hflip (augmentation "vô hại" phổ biến nhất):
                        augment=True, augment_scale_crop=False, augment_rotate=False,
                        augment_color_jitter=False, augment_synthetic_occlusion=False
                  - Full augmentation (bản cải tiến của bạn, mặc định cũ):
                        augment=True   (không override gì thêm)
            filter_degenerate (bool): Lọc bỏ region có amodal mask diện tích ~0 (annotation lỗi).
                Chỉ áp dụng cho mode "cocoa*". Nên luôn để True.
            filter_fully_occluded (bool): Lọc bỏ region có visible mask diện tích ~0
                (vật thể bị che hoàn toàn — không có tín hiệu M_v nào cho model).
                Mặc định False vì đây là sample hợp lệ, không phải lỗi dữ liệu.
            min_area_px (int): Ngưỡng số pixel tối thiểu (trên canvas polygon, không phải
                ảnh gốc) để coi một mask là "không rỗng".
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.coco_img_dir = coco_img_dir
        self.samples = []

        base_augment = mode in ("pix2gestalt", "cocoa_train") if augment is None else augment
        # Từng loại augmentation: nếu override=None thì kế thừa base_augment
        self.aug_hflip = base_augment if augment_hflip is None else augment_hflip
        self.aug_scale_crop = base_augment if augment_scale_crop is None else augment_scale_crop
        self.aug_rotate = base_augment if augment_rotate is None else augment_rotate
        self.aug_color_jitter = base_augment if augment_color_jitter is None else augment_color_jitter
        self.aug_synthetic_occlusion = (
            base_augment if augment_synthetic_occlusion is None else augment_synthetic_occlusion
        )
        # Giữ lại self.augment để không phá code cũ nào đang đọc thuộc tính này
        self.augment = base_augment

        # ==========================================
        # CHẾ ĐỘ 1: TOY DATASET
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
        # CHẾ ĐỘ 3: COCOA
        # ==========================================
        elif self.mode.startswith("cocoa"):
            print(f"⏳ Đang phân tích đồ thị JSON của {self.mode.upper()} ...")
            if not self.coco_img_dir:
                raise ValueError("LỖI: Chế độ COCOA yêu cầu phải truyền biến 'coco_img_dir'.")

            print("💾 Đang lập chỉ mục thư mục ảnh phẳng...")
            self.available_images = set(os.listdir(self.coco_img_dir))

            with open(self.data_dir, 'r') as f:
                cocoa_data = json.load(f)

            skipped_images_count = 0
            skipped_objects_count = 0
            skipped_degenerate_count = 0
            skipped_fully_occluded_count = 0
            candidate_samples = []

            for ann in cocoa_data['annotations']:
                img_filename = ann['url'].split('/')[-1]

                if img_filename not in self.available_images:
                    skipped_images_count += 1
                    skipped_objects_count += len([r for r in ann['regions'] if r.get('isStuff', 0) != 1])
                    continue

                depth_constraints = ann.get('depth_constraint', '')
                occluders_of = {}

                if depth_constraints:
                    pairs = depth_constraints.split(',')
                    for pair in pairs:
                        if '-' in pair:
                            front, back = pair.split('-')
                            front, back = int(front), int(back)
                            occluders_of.setdefault(back, []).append(front)

                regions = ann['regions']
                for region in regions:
                    if region.get('isStuff', 0) == 1:
                        continue

                    region_order = region.get('order')
                    if region_order is None:
                        continue

                    seg = region.get('segmentation')
                    if not seg or len(seg) == 0:
                        # annotation rỗng ngay từ gốc -> luôn loại, không tính vào thống kê filter riêng
                        skipped_degenerate_count += 1
                        continue

                    candidate_samples.append({
                        'img_filename': img_filename,
                        'region': region,
                        'region_order': region_order,
                        'all_regions': regions,
                        'occluders': occluders_of.get(region_order, []),
                    })

            # ------------------------------------------------------------
            # FIX #3: lọc degenerate / fully-occluded bằng diện tích polygon,
            # KHÔNG decode ảnh gốc -> nhanh, chỉ dùng toạ độ polygon.
            # ------------------------------------------------------------
            if filter_degenerate or filter_fully_occluded:
                for s in candidate_samples:
                    try:
                        amodal_area, visible_area = self._fast_areas(
                            s['region']['segmentation'],
                            self._collect_occluder_polys(s['occluders'], s['all_regions']),
                        )
                    except Exception:
                        # Polygon lỗi (vd toạ độ NaN, format lạ) -> coi như degenerate, loại bỏ
                        skipped_degenerate_count += 1
                        continue

                    if filter_degenerate and amodal_area < min_area_px:
                        skipped_degenerate_count += 1
                        continue
                    if filter_fully_occluded and visible_area < min_area_px:
                        skipped_fully_occluded_count += 1
                        continue

                    self.samples.append(s)
            else:
                self.samples = candidate_samples

            self.num_samples = len(self.samples)
            print(f"✅ Đã giải mã thành công {self.num_samples} vật thể từ COCOA.")
            if skipped_images_count > 0:
                print(f"⚠️ [RESEARCH MODE] Đã bỏ qua {skipped_images_count} ảnh bị thiếu trên ổ đĩa "
                      f"({skipped_objects_count} vật thể bị loại).")
            if skipped_degenerate_count > 0:
                print(f"⚠️ Đã lọc {skipped_degenerate_count} region degenerate "
                      f"(annotation rỗng/vỡ, amodal area < {min_area_px}px).")
            if filter_fully_occluded and skipped_fully_occluded_count > 0:
                print(f"⚠️ Đã lọc {skipped_fully_occluded_count} region bị che hoàn toàn "
                      f"(visible area < {min_area_px}px) — filter_fully_occluded=True.")
            elif skipped_fully_occluded_count == 0 and not filter_fully_occluded:
                # chỉ để tham khảo, không lọc gì cả -> đếm nhanh và báo cho biết tỉ lệ
                pass
        else:
            raise ValueError(f"Chế độ mode='{self.mode}' không được hỗ trợ!")

    # ==========================================
    # HELPERS cho việc lọc nhanh (không decode ảnh)
    # ==========================================
    @staticmethod
    def _collect_occluder_polys(occluder_orders, all_regions):
        polys = []
        for occ_order in occluder_orders:
            for r in all_regions:
                if r.get('order') == occ_order:
                    seg = r.get('segmentation')
                    if seg:
                        polys.append(seg)
                    break
        return polys

    @staticmethod
    def _poly_to_points_list(poly):
        """Chuẩn hoá polygon (dạng flat list hoặc list-of-list) thành list các (N,2) array."""
        if isinstance(poly[0], list):
            return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in poly]
        return [np.array(poly, dtype=np.float32).reshape(-1, 2)]

    def _fast_areas(self, amodal_poly, occluder_polys):
        """
        Tính diện tích amodal & visible bằng rasterize trên canvas nhỏ = bounding box
        của toàn bộ polygon liên quan (KHÔNG cần đọc ảnh gốc -> nhanh).
        Trả về (amodal_area_px, visible_area_px) trên canvas cục bộ này (không phải
        trên ảnh gốc / ảnh đã resize — chỉ dùng để so sánh với min_area_px làm ngưỡng
        rỗng/không rỗng).
        """
        amodal_pts_list = self._poly_to_points_list(amodal_poly)
        occ_pts_lists = [self._poly_to_points_list(p) for p in occluder_polys]

        all_pts = np.concatenate(amodal_pts_list + [p for lst in occ_pts_lists for p in lst], axis=0)
        min_xy = all_pts.min(axis=0) - 2
        max_xy = all_pts.max(axis=0) + 2
        w = max(int(np.ceil(max_xy[0] - min_xy[0])), 1)
        h = max(int(np.ceil(max_xy[1] - min_xy[1])), 1)

        def rasterize(pts_list):
            mask = np.zeros((h, w), dtype=np.uint8)
            polys_int = [(pts - min_xy).astype(np.int32) for pts in pts_list]
            cv2.fillPoly(mask, polys_int, 1)
            return mask

        amodal_mask = rasterize(amodal_pts_list)
        occ_mask = np.zeros((h, w), dtype=np.uint8)
        for pts_list in occ_pts_lists:
            occ_mask = np.logical_or(occ_mask, rasterize(pts_list)).astype(np.uint8)

        visible_mask = amodal_mask.copy()
        visible_mask[occ_mask == 1] = 0
        return int(amodal_mask.sum()), int(visible_mask.sum())

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
        if self.aug_hflip and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, ...])
            m_v = np.ascontiguousarray(m_v[:, ::-1])
            m_a = np.ascontiguousarray(m_a[:, ::-1])
        return img, m_v, m_a

    def _maybe_scale_crop(self, img, m_v, m_a):
        if not self.aug_scale_crop:
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
        if not self.aug_color_jitter:
            return img
        img = img.astype(np.float32)
        brightness = np.random.uniform(0.85, 1.15)
        contrast = np.random.uniform(0.85, 1.15)
        mean_val = img.mean()
        img = (img - mean_val) * contrast + mean_val
        img = img * brightness
        return np.clip(img, 0, 255)

    def _maybe_rotate(self, img, m_v, m_a, max_angle=15):
        if not self.aug_rotate or np.random.rand() < 0.5:
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
        if not self.aug_synthetic_occlusion or np.random.rand() > prob:
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
            img_path = os.path.join(self.coco_img_dir, img_filename)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img.shape[:2]

            target_poly = sample['region']['segmentation']
            amodal_mask = self._poly_to_mask(target_poly, h_orig, w_orig)

            occluder_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            all_regions = sample['all_regions']

            for occ_order in sample['occluders']:
                for r in all_regions:
                    if r.get('order') == occ_order:
                        occ_poly = r['segmentation']
                        occ_mask = self._poly_to_mask(occ_poly, h_orig, w_orig)
                        occluder_mask = np.logical_or(occluder_mask, occ_mask).astype(np.uint8)
                        break

            modal_mask = amodal_mask.copy()
            modal_mask[occluder_mask == 1] = 0

            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            m_v = cv2.resize(modal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            m_a = cv2.resize(amodal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            img, m_v, m_a = self._augment_all(img, m_v, m_a)

            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            m_v = m_v.astype(np.float32)
            m_a = m_a.astype(np.float32)

            img = np.transpose(img, (2, 0, 1))
            m_v = np.expand_dims(m_v, axis=0)
            m_a = np.expand_dims(m_a, axis=0)

            return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a)