import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

# Import các mảnh ghép hoàn hảo mà chúng ta đã xây dựng
from src.dataset import AmodalDataset
from src.utils.geometry import get_distance_map
from src.schedulers.da_ust import GaussianDAUSTScheduler
from src.models.pcn_extractor import PCNExtractor
from src.models.denoising_net import DenoisingNetwork, WeightedBCELoss

# ==========================================
# CẤU HÌNH SIÊU THAM SỐ (HYPERPARAMETERS)
# ==========================================
# Lấy từ file cấu hình của bạn, hoặc cấu hình cứng ở đây cho dễ chạy Kaggle
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
IMAGE_SIZE = (256, 256)
TIMESTEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# KHỞI TẠO MÔ HÌNH VÀ CÁC THÀNH PHẦN
# ==========================================
def build_system():
    print(f"🚀 Khởi động hệ thống trên thiết bị: {DEVICE}")
    
    # 1. Khởi tạo Dataset (Hiện tại đang dùng Toy Dataset để test, đổi mode sau)
    print("⏳ Đang tải dữ liệu...")
    dataset = AmodalDataset(data_dir="", mode="toy", image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Khởi tạo Trái tim toán học: DA-UST Scheduler
    print("⏳ Đang nạp thuật toán Distance-Aware UST...")
    scheduler = GaussianDAUSTScheduler(num_train_timesteps=TIMESTEPS, sigma=5.0)
    
    # 3. Khởi tạo PCN (Dùng PVTv2-B2, chế độ use_hf=False để lấy 4 tầng Kim tự tháp)
    print("⏳ Đang tải PVTv2-B2 Backbone...")
    pcn = PCNExtractor(model_name='pvt_v2_b2', pretrained=True).to(DEVICE)
    
    # 4. Khởi tạo Mạng Khử nhiễu Custom DN
    print("⏳ Đang lắp ráp Denoising Network...")
    # Lấy số channels thực tế từ PVTv2-B2 để đưa vào LE module
    pcn_channels = pcn.feature_channels # Thường là [64, 128, 320, 512]
    dn = DenoisingNetwork(pcn_channels=pcn_channels, fuse_channels=256).to(DEVICE)
    
    # 5. Hàm Loss và Optimizer
    criterion = WeightedBCELoss().to(DEVICE)
    # Tối ưu hóa cả 2 mạng cùng lúc
    optimizer = optim.AdamW(list(pcn.parameters()) + list(dn.parameters()), lr=LEARNING_RATE)
    
    return dataloader, scheduler, pcn, dn, criterion, optimizer

# ==========================================
# VÒNG LẶP HUẤN LUYỆN CHÍNH (TRAINING LOOP)
# ==========================================
def train():
    dataloader, scheduler, pcn, dn, criterion, optimizer = build_system()
    
    print("\n" + "="*50)
    print("🔥 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN 🔥")
    print("="*50)
    
    for epoch in range(NUM_EPOCHS):
        pcn.train()
        dn.train()
        epoch_loss = 0.0
        
        # Thanh tiến trình đẹp mắt giống trainer của tác giả
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in pbar:
            # 1. Đưa dữ liệu lên GPU
            I, M_v, M_a = batch
            I, M_v, M_a = I.to(DEVICE), M_v.to(DEVICE), M_a.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 2. Sinh ngẫu nhiên bước thời gian t cho batch
            batch_size = I.shape[0]
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
            
            # 3. Áp dụng DA-UST (Forward Process)
            # - Tính toán ma trận khoảng cách từ M_v
            distance_map = get_distance_map(M_v)
            # - Tạo nhiễu x_t lên M_a dựa trên khoảng cách
            x_t = scheduler.add_noise(M_a, distance_map, t)
            
            # 4. Trích xuất Đặc trưng Đa cấp độ (PCN)
            # Chạy PVTv2 để lấy [F1, F2, F3, F4]
            pyramid_features = pcn(I, M_v, x_t, t, use_hf=False)
            
            # 5. Khử nhiễu và Dự đoán (DN)
            # Mạng DN nhận x_t, thời gian t, và điều kiện F1-F4 để dự đoán x_hat_0
            x_hat_0_logits = dn(x_t, t, pyramid_features)
            
            # 6. Tính Loss và Cập nhật trọng số
            # So sánh Logits dự đoán với M_a gốc (Ground Truth)
            loss = criterion(x_hat_0_logits, M_a)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pcn.parameters()) + list(dn.parameters()), 1.0)
            optimizer.step()
            
            # 7. Cập nhật giao diện
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # --- Lưu Checkpoint ---
        avg_loss = epoch_loss / len(dataloader)
        print(f"📈 Epoch {epoch+1} hoàn tất | Trung bình Loss: {avg_loss:.5f}")
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            ckpt_path = os.path.join(SAVE_DIR, f"conddiff_amo_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'pcn_state_dict': pcn.state_dict(),
                'dn_state_dict': dn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"💾 Đã lưu Checkpoint tại: {ckpt_path}")

if __name__ == "__main__":
    # Chỉ gọi hàm train khi chạy trực tiếp file này
    train()