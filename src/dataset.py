import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class AmodalDataset(Dataset):
    def __init__(self, data_dir, mode="toy", image_size=(256, 256)):
        """
        Khởi tạo Dataloader hỗ trợ linh hoạt các chiến lược huấn luyện.
        
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa data.
            mode (str): "toy" (Sinh dữ liệu ảo để debug), 
                        "cocoa" (Đọc format COCOA), 
                        "pix2gestalt" (Đọc format 120GB).
            image_size (tuple): Kích thước chuẩn hóa để đưa vào mạng U-Net.
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        
        if self.mode == "toy":
            print("🚀 Đang chạy chế độ TOY DATASET (Sinh dữ liệu giả lập trên RAM) để test code!")
            self.num_samples = 100 # Tạo ra 100 ảnh ảo để chạy qua vài epoch
            
        elif self.mode == "cocoa":
            # Logic quét thư mục của COCOA (Giả định cấu trúc thư mục)
            self.image_paths = sorted([os.path.join(data_dir, "images", f) for f in os.listdir(os.path.join(data_dir, "images"))])
            self.num_samples = len(self.image_paths)
            print(f"📁 Đã load {self.num_samples} mẫu từ tập COCOA.")
            
        elif self.mode == "pix2gestalt":
            print("⏳ Khởi tạo chế độ Pix2Gestalt, đang quét dữ liệu...")
            # Khai báo các đường dẫn thư mục con dựa theo cấu trúc của pix2gestalt
            self.occlusion_dir = os.path.join(data_dir, 'occlusion')
            self.visible_mask_dir = os.path.join(data_dir, 'visible_object_mask')
            self.whole_mask_dir = os.path.join(data_dir, 'whole_mask')
            
            # Lấy danh sách tên file hợp lệ (kiểm tra thư mục occlusion làm chuẩn)
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
            self.image_names = sorted([
                f for f in os.listdir(self.occlusion_dir) 
                if f.lower().endswith(valid_exts)
            ])
            
            self.num_samples = len(self.image_names)
            print(f"✅ Đã tìm thấy {self.num_samples} mẫu dữ liệu Pix2Gestalt.")

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
            
        elif self.mode == "cocoa":
            # TODO: Xử lý COCOA
            pass
            
        elif self.mode == "pix2gestalt":
            img_name = self.image_names[idx]
            
            # Khởi tạo đường dẫn cụ thể của từng file
            img_path = os.path.join(self.occlusion_dir, img_name)
            m_v_path = os.path.join(self.visible_mask_dir, img_name)
            m_a_path = os.path.join(self.whole_mask_dir, img_name)
            
            # Đọc ảnh RGB chứa vật bị che (Occlusion Image)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Chuyển từ BGR (OpenCV) sang RGB
            
            # Đọc Mask (Đọc dưới dạng Grayscale)
            # Dùng cv2.IMREAD_UNCHANGED (-1) giống file simple.py của tác giả
            m_v = cv2.imread(m_v_path, cv2.IMREAD_UNCHANGED)
            m_a = cv2.imread(m_a_path, cv2.IMREAD_UNCHANGED)
            
            # Resize ảnh và mask về kích thước chuẩn (ví dụ 256x256)
            # Lưu ý: Mask dùng INTER_NEAREST để không làm nhòe các giá trị nhị phân (0, 1)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            m_v = cv2.resize(m_v, self.image_size, interpolation=cv2.INTER_NEAREST)
            m_a = cv2.resize(m_a, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # Chuẩn hóa về [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Mask của Pix2Gestalt thường có giá trị 0-255, ta quy về nhị phân 0-1
            m_v = (m_v > 127).astype(np.float32)
            m_a = (m_a > 127).astype(np.float32)
            
            # Chuyển Data Layout cho PyTorch
            img = np.transpose(img, (2, 0, 1))           # (H, W, C) -> (C, H, W)
            m_v = np.expand_dims(m_v, axis=0)            # (H, W) -> (1, H, W)
            m_a = np.expand_dims(m_a, axis=0)            # (H, W) -> (1, H, W)
            
            return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a)

# Khối Test nhanh
if __name__ == "__main__":
    # Test thử chế độ Toy Dataset
    toy_dataset = AmodalDataset(data_dir="", mode="toy")
    img, m_v, m_a = toy_dataset[0]
    
    print(f"Kích thước Ảnh (I): {img.shape}")
    print(f"Kích thước Modal Mask (M_v): {m_v.shape}")
    print(f"Kích thước Amodal Mask (M_a): {m_a.shape}")
    print("Dataloader đã sẵn sàng bơm máu cho hệ thống!")