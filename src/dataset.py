import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json

class AmodalDataset(Dataset):
    def __init__(self, data_dir, mode="toy", image_size=(256, 256), coco_img_dir=None):
        """
        Trái tim xử lý dữ liệu của dự án CondDiff-AMO.
        
        Args:
            data_dir (str): Đường dẫn đến thư mục dữ liệu (hoặc file .json nếu là COCOA).
            mode (str): Chế độ chạy ("toy", "pix2gestalt", "cocoa_train", "cocoa_val", "cocoa_test").
            image_size (tuple): Kích thước chuẩn hóa đầu vào (H, W).
            coco_img_dir (str): Đường dẫn đến thư mục ảnh gốc COCO (Bắt buộc cho COCOA).
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size 
        self.coco_img_dir = coco_img_dir
        self.samples = [] 
        
        # ==========================================
        # CHẾ ĐỘ 1: TOY DATASET (Sinh dữ liệu giả lập)
        # ==========================================
        if self.mode == "toy":
            print("🚀 Đang khởi tạo Toy Dataset (Dữ liệu giả lập hình khối)...")
            self.num_samples = 100 
            
        # ==========================================
        # CHẾ ĐỘ 2: PIX2GESTALT (Dữ liệu thư mục truyền thống)
        # ==========================================
        elif self.mode == "pix2gestalt":
            print("⏳ Đang quét và đối chiếu chéo dữ liệu Pix2Gestalt ...")
            self.occlusion_dir = os.path.join(data_dir, 'occlusion')
            self.visible_mask_dir = os.path.join(data_dir, 'visible_object_mask')
            self.whole_mask_dir = os.path.join(data_dir, 'whole_mask')
            
            # Lọc chéo để đảm bảo ID tồn tại ở cả 3 thư mục
            occ_ids = {f.split('_')[0] for f in os.listdir(self.occlusion_dir) if f.endswith('.png')}
            vis_ids = {f.split('_')[0] for f in os.listdir(self.visible_mask_dir) if f.endswith('.png')}
            whole_ids = {f.split('_')[0] for f in os.listdir(self.whole_mask_dir) if f.endswith('.png')}
            
            valid_ids = occ_ids & vis_ids & whole_ids
            self.base_ids = sorted(list(valid_ids))
            self.num_samples = len(self.base_ids)
            print(f"✅ Đã tìm thấy {self.num_samples} mẫu hợp lệ.")

        # ==========================================
        # CHẾ ĐỘ 3: COCOA (Hệ sinh thái chuẩn Benchmark)
        # ==========================================
        elif self.mode.startswith("cocoa"):
            print(f"⏳ Đang phân tích đồ thị JSON của {self.mode.upper()} ...")
            if not self.coco_img_dir:
                raise ValueError("LỖI: Chế độ COCOA yêu cầu phải truyền biến 'coco_img_dir'.")
            
            # --- BỘ QUÉT 3 NGÃ: TRAIN, VAL VÀ TEST ---
            print("💾 Đang lập chỉ mục ảnh COCO 2017 để tăng tốc xử lý...")
            self.train2017_set = set(os.listdir(os.path.join(self.coco_img_dir, "train2017")))
            self.val2017_set = set(os.listdir(os.path.join(self.coco_img_dir, "val2017")))
            
            # Bổ sung quét thêm thư mục test2017
            test_dir = os.path.join(self.coco_img_dir, "test2017")
            if os.path.exists(test_dir):
                self.test2017_set = set(os.listdir(test_dir))
            else:
                self.test2017_set = set()
            
            # CHỐT CHẶN BẮT LỖI ĐƯỜNG DẪN RỖNG:
            print(f"📊 Đã tìm thấy: {len(self.train2017_set)} ảnh Train, {len(self.val2017_set)} ảnh Val và {len(self.test2017_set)} ảnh Test.")
            if len(self.train2017_set) == 0:
                print("⚠️ CẢNH BÁO ĐỎ: Thư mục ảnh hoàn toàn trống rỗng! Hãy kiểm tra lại đường dẫn COCO_IMG_DIR.")

            with open(self.data_dir, 'r') as f:
                cocoa_data = json.load(f)
            
            # Quét từng bức ảnh trong file JSON (Giữ nguyên logic phân tích của bạn)
            for ann in cocoa_data['annotations']:
                img_filename = ann['url'].split('/')[-1] 
                
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

                regions = ann['regions']
                for region in regions:
                    if region.get('isStuff', 0) == 1: 
                        continue
                    region_order = region.get('order')
                    if region_order is not None:
                        self.samples.append({
                            'img_filename': img_filename, # Vẫn lưu tên gốc 2014 từ JSON phục vụ map ID
                            'region': region,
                            'region_order': region_order,
                            'all_regions': regions,
                            'occluders': occluders_of.get(region_order, [])
                        })
                    
            self.num_samples = len(self.samples)
            print(f"✅ Đã giải mã thành công {self.num_samples} vật thể từ COCOA tương thích COCO 2017.")

    def __len__(self):
        return self.num_samples

    # ==========================================
    # CÁC HÀM XỬ LÝ LÕI (CORE FUNCTIONS)
    # ==========================================
    def _generate_toy_data(self):
        """Hàm tự sinh: Hình tròn (Amodal) bị che bởi Hình vuông (Occluder)"""
        h, w = self.image_size
        image = np.zeros((h, w, 3), dtype=np.uint8)
        amodal_mask = np.zeros((h, w), dtype=np.uint8)
        
        center = (w // 2, h // 2)
        radius = h // 4
        cv2.circle(amodal_mask, center, radius, 1, -1)
        
        occluder = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(occluder, center, (w, h), 1, -1)
        
        # Modal = Amodal - Occluder
        modal_mask = amodal_mask.copy()
        modal_mask[occluder == 1] = 0
        
        image[amodal_mask == 1] = [0, 255, 0] 
        image[occluder == 1] = [255, 0, 0]    
        
        image = image.astype(np.float32) / 255.0
        modal_mask = modal_mask.astype(np.float32)
        amodal_mask = amodal_mask.astype(np.float32)
        
        image = np.transpose(image, (2, 0, 1))
        modal_mask = np.expand_dims(modal_mask, axis=0)
        amodal_mask = np.expand_dims(amodal_mask, axis=0)
        
        return torch.tensor(image), torch.tensor(modal_mask), torch.tensor(amodal_mask)

    def _poly_to_mask(self, polygons, h, w):
        """Chuyển đổi tọa độ JSON thành Ma trận Nhị phân"""
        mask = np.zeros((h, w), dtype=np.uint8)
        if isinstance(polygons[0], list):
            pts = [np.array(p, np.int32).reshape((-1, 2)) for p in polygons]
        else:
            pts = [np.array(polygons, np.int32).reshape((-1, 2))]
        cv2.fillPoly(mask, pts, 1)
        return mask

    # ==========================================
    # PIPELINE CUNG CẤP DỮ LIỆU
    # ==========================================
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
            
            img = img.astype(np.float32) / 255.0
            m_v = (m_v > 127).astype(np.float32)
            m_a = (m_a > 127).astype(np.float32)
            
            img = np.transpose(img, (2, 0, 1))
            m_v = np.expand_dims(m_v, axis=0)
            m_a = np.expand_dims(m_a, axis=0)
            
            return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a)
            
        elif self.mode.startswith("cocoa"):
            sample = self.samples[idx]
            old_filename = sample['img_filename'] 
            
            # ÉP FORMAT TÊN FILE TỪ 2014 SANG CHUẨN 2017
            coco2017_filename = old_filename.split('_')[-1] 
            
            # BỘ ĐỊNH TUYẾN 3 NGÃ THÔNG MINH
            if coco2017_filename in self.train2017_set:
                img_path = os.path.join(self.coco_img_dir, "train2017", coco2017_filename)
            elif coco2017_filename in self.val2017_set:
                img_path = os.path.join(self.coco_img_dir, "val2017", coco2017_filename)
            elif coco2017_filename in self.test2017_set:
                img_path = os.path.join(self.coco_img_dir, "test2017", coco2017_filename)
            else:
                raise FileNotFoundError(f"🚨 Không tìm thấy ảnh {coco2017_filename} trong cả 3 tập train, val và test của COCO 2017!")
                
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img.shape[:2]
            
            # ==========================================
            # 2. DỰNG AMODAL MASK
            # ==========================================
            target_poly = sample['region']['segmentation']
            amodal_mask = self._poly_to_mask(target_poly, h_orig, w_orig)
            
            # ==========================================
            # 3. DỰNG OCCLUDER MASK (Kẻ che khuất)
            # ==========================================
            occluder_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            all_regions = sample['all_regions']
            
            for occ_order in sample['occluders']:
                for r in all_regions:
                    if r.get('order') == occ_order:
                        occ_poly = r['segmentation']
                        occ_mask = self._poly_to_mask(occ_poly, h_orig, w_orig)
                        occluder_mask = np.logical_or(occluder_mask, occ_mask).astype(np.uint8)
                        break 
                        
            # ==========================================
            # 4. TÍNH TOÁN MODAL MASK
            # ==========================================
            modal_mask = amodal_mask.copy()
            modal_mask[occluder_mask == 1] = 0
            
            # ==========================================
            # 5. TIỀN XỬ LÝ & TRẢ VỀ TENSOR
            # ==========================================
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            m_v = cv2.resize(modal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            m_a = cv2.resize(amodal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            img = img.astype(np.float32) / 255.0
            m_v = m_v.astype(np.float32)
            m_a = m_a.astype(np.float32)
            
            img = np.transpose(img, (2, 0, 1))
            m_v = np.expand_dims(m_v, axis=0)
            m_a = np.expand_dims(m_a, axis=0)
            
            return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a)