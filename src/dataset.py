import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class AmodalDataset(Dataset):
    def __init__(self, data_dir, mode="toy", image_size=(256, 256)):
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.samples = [] # Danh sách lưu trữ các bộ 3 đường dẫn hợp lệ
        
        if self.mode == "toy":
            print("🚀 Đang chạy chế độ TOY DATASET!")
            self.num_samples = 100 
            
        elif self.mode == "pix2gestalt":
            print("⏳ Khởi tạo chế độ Pix2Gestalt, đang quét dữ liệu...")
            self.occlusion_dir = os.path.join(data_dir, 'occlusion')
            self.visible_mask_dir = os.path.join(data_dir, 'visible_object_mask')
            self.whole_mask_dir = os.path.join(data_dir, 'whole_mask')
            
            # Quét tất cả các file trong thư mục occlusion
            all_occlusion_files = os.listdir(self.occlusion_dir)
            missing_files_count = 0
            
            for filename in all_occlusion_files:
                if not filename.endswith('.png'): 
                    continue
                
                # 1. Tách lấy mã ID. Ví dụ: '10000015_occlusion.png' -> '10000015'
                base_id = filename.split('_')[0]
                
                # 2. Xây dựng đường dẫn tuyệt đối cho bộ 3 file dựa trên format của Kaggle
                img_path = os.path.join(self.occlusion_dir, f"{base_id}_occlusion.png")
                m_v_path = os.path.join(self.visible_mask_dir, f"{base_id}_visible_mask.png")
                m_a_path = os.path.join(self.whole_mask_dir, f"{base_id}_whole_mask.png")
                
                # 3. Sanity Check: Kiểm tra xem các file Mask có thực sự tồn tại trên ổ cứng không
                if os.path.exists(m_v_path) and os.path.exists(m_a_path):
                    # Nếu đủ mặt anh tài, cho vào danh sách huấn luyện
                    self.samples.append({
                        'img_path': img_path,
                        'm_v_path': m_v_path,
                        'm_a_path': m_a_path
                    })
                else:
                    # Ghi nhận có file bị thiếu/lỗi tải về
                    missing_files_count += 1
            
            self.num_samples = len(self.samples)
            print(f"✅ Quét thành công {self.num_samples} mẫu dữ liệu hợp lệ.")
            if missing_files_count > 0:
                print(f"⚠️ Đã tự động bỏ qua {missing_files_count} mẫu bị thiếu file mask.")

    def __len__(self):
        return self.num_samples

    def _generate_toy_data(self):
        # ... (Phần logic Toy Data giữ nguyên như cũ) ...
        pass

    def __getitem__(self, idx):
        if self.mode == "toy":
            return self._generate_toy_data()
            
        elif self.mode == "pix2gestalt":
            # Lấy bộ 3 đường dẫn chuẩn xác
            paths = self.samples[idx]
            
            img = cv2.imread(paths['img_path'])
            m_v = cv2.imread(paths['m_v_path'], cv2.IMREAD_UNCHANGED)
            m_a = cv2.imread(paths['m_a_path'], cv2.IMREAD_UNCHANGED)
            
            # Chốt chặn cuối cùng (Đề phòng file có tồn tại nhưng bị hỏng nội dung)
            if img is None or m_v is None or m_a is None:
                raise ValueError(f"File bị hỏng không thể đọc bằng OpenCV tại ID: {paths['img_path']}")
            
            # Đổi hệ màu BGR sang RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            m_v = cv2.resize(m_v, self.image_size, interpolation=cv2.INTER_NEAREST)
            m_a = cv2.resize(m_a, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # Chuẩn hóa về [0, 1]
            img = img.astype(np.float32) / 255.0
            m_v = (m_v > 127).astype(np.float32)
            m_a = (m_a > 127).astype(np.float32)
            
            # Chuyển Data Layout cho PyTorch (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            m_v = np.expand_dims(m_v, axis=0)
            m_a = np.expand_dims(m_a, axis=0)
            
            return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a)