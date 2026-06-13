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
            # Logic quét thư mục của Pix2Gestalt
            # (Bạn sẽ điền logic tương ứng khi tải xong 120GB)
            self.num_samples = 0 
            print("⏳ Chế độ Pix2Gestalt đã sẵn sàng, chờ mount dữ liệu 120GB.")

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
        
        # 5. Đổ màu cho Ảnh RGB để hiển thị (Hình tròn màu Xanh lá, Hình vuông che màu Đỏ)
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
            # Ở đây bạn sẽ viết lệnh cv2.imread để đọc thật từ ổ cứng
            # Hiện tại cứ dùng hàm toy để tránh lỗi khi chưa có thư mục thật
            pass
            
        elif self.mode == "pix2gestalt":
            # Logic đọc file 120GB
            pass

# Khối Test nhanh
if __name__ == "__main__":
    # Test thử chế độ Toy Dataset
    toy_dataset = AmodalDataset(data_dir="", mode="toy")
    img, m_v, m_a = toy_dataset[0]
    
    print(f"Kích thước Ảnh (I): {img.shape}")
    print(f"Kích thước Modal Mask (M_v): {m_v.shape}")
    print(f"Kích thước Amodal Mask (M_a): {m_a.shape}")
    print("Dataloader đã sẵn sàng bơm máu cho hệ thống!")