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
            print("⏳ Đang quét và đối chiếu chéo dữ liệu ...")
            self.occlusion_dir = os.path.join(data_dir, 'occlusion')
            self.visible_mask_dir = os.path.join(data_dir, 'visible_object_mask')
            self.whole_mask_dir = os.path.join(data_dir, 'whole_mask')
            
            # Dùng Set Comprehension để lấy mã ID từ cả 3 thư mục
            occ_ids = {f.split('_')[0] for f in os.listdir(self.occlusion_dir) if f.endswith('.png')}
            vis_ids = {f.split('_')[0] for f in os.listdir(self.visible_mask_dir) if f.endswith('.png')}
            whole_ids = {f.split('_')[0] for f in os.listdir(self.whole_mask_dir) if f.endswith('.png')}
            
            # Phép Giao Toán Học: Chỉ lấy những ID tồn tại ở CẢ 3 THƯ MỤC
            valid_ids = occ_ids & vis_ids & whole_ids
            
            self.base_ids = sorted(list(valid_ids))
            self.num_samples = len(self.base_ids)
            
            print(f"Có {self.num_samples} mẫu hợp lệ 100%.")

    def __len__(self):
        return self.num_samples

    def _generate_toy_data(self):
        """Hàm tự động sinh ra một vật thể tròn bị che khuất bởi vật thể vuông."""
        h, w = self.image_size
        
        # 1. Tạo Ảnh RGB nền đen
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 2. Tạo Amodal Mask (M_a): Đích đến là một hình tròn hoàn hảo ở giữa
        amodal_mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = h // 4
        cv2.circle(amodal_mask, center, radius, 1, -1)
        
        # 3. Tạo Vật thể che khuất (Occluder): Một hình vuông đè lên góc phải dưới của hình tròn
        occluder = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(occluder, center, (w, h), 1, -1)
        
        # 4. Tạo Modal Mask (M_v): Phần hình tròn KHÔNG bị hình vuông che
        modal_mask = amodal_mask.copy()
        modal_mask[occluder == 1] = 0
        
        # 5. Đổ màu cho Ảnh RGB để hiển thị
        image[amodal_mask == 1] = [0, 255, 0] # Vật thể chính
        image[occluder == 1] = [255, 0, 0]    # Vật che khuất
        
        # Chuẩn hóa về [0, 1] cho Neural Network
        image = image.astype(np.float32) / 255.0
        modal_mask = modal_mask.astype(np.float32)
        amodal_mask = amodal_mask.astype(np.float32)
        
        # Đổi trục (H, W, C) -> (C, H, W) cho PyTorch
        image = np.transpose(image, (2, 0, 1))
        modal_mask = np.expand_dims(modal_mask, axis=0)
        amodal_mask = np.expand_dims(amodal_mask, axis=0)
        
        return torch.tensor(image), torch.tensor(modal_mask), torch.tensor(amodal_mask)

    def __getitem__(self, idx):
        if self.mode == "toy":
            return self._generate_toy_data()
            
        elif self.mode == "pix2gestalt":
            # Lấy bộ 3 đường dẫn chuẩn xác
            base_id = self.base_ids[idx]
            
            # Tự động sinh đường dẫn ngay lúc bốc data
            img_path = os.path.join(self.occlusion_dir, f"{base_id}_occlusion.png")
            m_v_path = os.path.join(self.visible_mask_dir, f"{base_id}_visible_mask.png")
            m_a_path = os.path.join(self.whole_mask_dir, f"{base_id}_whole_mask.png")
            
            img = cv2.imread(img_path)
            m_v = cv2.imread(m_v_path)
            m_a = cv2.imread(m_a_path)
            
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