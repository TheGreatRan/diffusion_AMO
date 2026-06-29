import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import shutil
import gc

# Ép matplotlib chạy ở chế độ Headless
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Import các mảnh ghép
from src.dataset import AmodalDataset
from src.utils.geometry import get_distance_map
from src.models.pcn_extractor import PCNExtractor
from src.models.denoising_net import DenoisingNetwork, WeightedBCELoss, WeightedIoULoss

# IMPORT CẢ 3 CHIẾN LƯỢC ĐỂ CHẠY ABLATION STUDY
from src.schedulers.da_ust import BaselineUSTScheduler, ClampedDAUSTScheduler, ExponentialDAUSTScheduler

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & SIÊU THAM SỐ
# ==========================================
COCO_IMG_DIR = "/kaggle/input/datasets/ralphsitinh/cocoa-image/cocoa_images_extracted"
COCOA_TRAIN_JSON = "/kaggle/input/datasets/ralphsitinh/coco-amodal-annotations/annotations/COCO_amodal_train2014.json"
COCOA_VAL_JSON = "/kaggle/input/datasets/ralphsitinh/coco-amodal-annotations/annotations/COCO_amodal_val2014.json"

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMAGE_SIZE = (256, 256)
TIMESTEPS = 1000
NUM_EPOCHS = 12        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "/kaggle/working/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_TRAINING = True      
MAX_TRAIN_HOURS = 11.5        

# Danh sách các chiến lược cần thử nghiệm
STRATEGIES_TO_RUN = ["exponential"]

# ==========================================
# HÀM TRỢ GIÚP: VẼ BIỂU ĐỒ (RIÊNG & CHUNG)
# ==========================================
def save_loss_plot_individual(history, save_path, strategy_name):
    """Vẽ biểu đồ riêng cho từng chiến lược"""
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_steps'], history['train_losses'], label='Train Loss', alpha=0.6, color='blue')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'[{strategy_name.upper()}] Train Loss (Steps)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    if len(history['epoch_train_losses']) > 0:
        epochs = range(1, len(history['epoch_train_losses']) + 1)
        plt.plot(epochs, history['epoch_train_losses'], label='Train', marker='o')
        plt.plot(epochs, history['epoch_val_losses'], label='Val', marker='s')
        plt.xlabel('Epochs')
        plt.title(f'[{strategy_name.upper()}] Train vs Val Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_combined_val_plot(all_histories, save_path):
    """Vẽ biểu đồ gộp so sánh Val Loss của tất cả chiến lược"""
    plt.figure(figsize=(10, 6))
    
    colors = {'baseline': 'red', 'clamped': 'orange', 'exponential': 'green'}
    markers = {'baseline': 'x', 'clamped': '^', 'exponential': 'o'}
    
    for strategy_name, history in all_histories.items():
        if len(history['epoch_val_losses']) > 0:
            epochs = range(1, len(history['epoch_val_losses']) + 1)
            plt.plot(epochs, history['epoch_val_losses'], 
                     label=f'{strategy_name.upper()}', 
                     color=colors.get(strategy_name, 'blue'),
                     marker=markers.get(strategy_name, 'o'),
                     linewidth=2, alpha=0.8)
            
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Ablation Study: So Sánh Hiệu Quả Xói Mòn Không Gian (Val Loss)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"📈 Đã cập nhật biểu đồ so sánh tổng hợp tại: {save_path}")

# ==========================================
# KHỞI TẠO HỆ THỐNG THEO TỪNG CHIẾN LƯỢC
# ==========================================
def build_system(strategy_name):
    print(f"\n" + "="*50)
    print(f"🚀 KHỞI TẠO HỆ THỐNG: CHIẾN LƯỢC [{strategy_name.upper()}]")
    print("="*50)
    
    train_dataset = AmodalDataset(data_dir=COCOA_TRAIN_JSON, mode="cocoa_train", image_size=IMAGE_SIZE, coco_img_dir=COCO_IMG_DIR)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    val_dataset = AmodalDataset(data_dir=COCOA_VAL_JSON, mode="cocoa_val", image_size=IMAGE_SIZE, coco_img_dir=COCO_IMG_DIR)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # 🎯 SWITCHER LỰA CHỌN SCHEDULER
    if strategy_name == "baseline":
        scheduler = BaselineUSTScheduler(num_train_timesteps=TIMESTEPS)
    elif strategy_name == "clamped":
        scheduler = ClampedDAUSTScheduler(num_train_timesteps=TIMESTEPS, sigma=5.0, beta_min=0.05)
    elif strategy_name == "exponential":
        scheduler = ExponentialDAUSTScheduler(num_train_timesteps=TIMESTEPS, gamma=5.0, beta_min=0.05)
    else:
        raise ValueError(f"Không nhận diện được chiến lược: {strategy_name}")
    
    pcn = PCNExtractor(model_name='pvt_v2_b2', pretrained=True).to(DEVICE)
    dn = DenoisingNetwork(pcn_channels=pcn.feature_channels, fuse_channels=256).to(DEVICE)
    
    bce_criterion = WeightedBCELoss().to(DEVICE)
    iou_criterion = WeightedIoULoss().to(DEVICE)
    
    optimizer = optim.AdamW([
        {'params': pcn.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': dn.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-2)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    return train_dataloader, val_dataloader, scheduler, pcn, dn, criterion, optimizer, lr_scheduler

# ==========================================
# HÀM ĐÁNH GIÁ MÔ HÌNH (GIỮ NGUYÊN)
# ==========================================
def validate(val_dataloader, scheduler, pcn, dn, criterion):
    pcn.eval()
    dn.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_dataloader:
            I, M_v, M_a = [x.to(DEVICE) for x in batch]
            
            batch_size = I.shape[0]
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
            distance_map = get_distance_map(M_v)
            x_t = scheduler.add_noise(M_a, distance_map, t)
            
            pyramid_features = pcn(I, M_v, x_t, t, use_hf=False)
            x_hat_0_logits = dn(x_t, t, pyramid_features)
            
            loss = criterion(x_hat_0_logits, M_a)
            val_loss += loss.item()
            
    return val_loss / len(val_dataloader)

# ==========================================
# MODULE HUẤN LUYỆN 1 CHIẾN LƯỢC ĐỘC LẬP
# ==========================================
def train_strategy(strategy_name, global_start_time, time_limit_seconds):
    train_dataloader, val_dataloader, scheduler, pcn, dn, criterion, optimizer, lr_scheduler = build_system(strategy_name)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    history = {
        'train_steps': [], 'train_losses': [],
        'epoch_train_losses': [], 'epoch_val_losses': []
    }
    
    # Định danh file tạ cho từng chiến lược
    latest_ckpt_path = os.path.join(SAVE_DIR, f"latest_ckpt_{strategy_name}.pth")
    best_ckpt_path = os.path.join(SAVE_DIR, f"best_ckpt_{strategy_name}.pth")
    indiv_plot_path = f"/kaggle/working/loss_chart_{strategy_name}.png"
    
    if RESUME_TRAINING and os.path.exists(latest_ckpt_path):
        print(f"♻️ Tìm thấy tiến trình cũ của [{strategy_name}]. Đang khôi phục...")
        checkpoint = torch.load(latest_ckpt_path, map_location=DEVICE)
        pcn.load_state_dict(checkpoint['pcn_state_dict'])
        dn.load_state_dict(checkpoint['dn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', history)
        global_step = checkpoint.get('global_step', start_epoch * len(train_dataloader))
        print(f"✅ Khôi phục thành công. Train tiếp từ Epoch {start_epoch + 1}...")
    else:
        global_step = 0

    if start_epoch >= NUM_EPOCHS:
        print(f"🎉 Chiến lược [{strategy_name}] đã hoàn thành 100% (đủ {NUM_EPOCHS} epochs). Bỏ qua huấn luyện.")
        return history

    for epoch in range(start_epoch, NUM_EPOCHS):
        if (time.time() - global_start_time) > time_limit_seconds:
            print(f"\n⏰ Sắp hết giờ Kaggle! Tạm ngắt chiến lược [{strategy_name}] để bảo toàn kết quả...")
            return history
            
        pcn.train()
        dn.train()
        train_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"[{strategy_name.upper()}] Ep {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            I, M_v, M_a = [x.to(DEVICE) for x in batch]
            
            optimizer.zero_grad()
            batch_size = I.shape[0]
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
            distance_map = get_distance_map(M_v)
            
            x_t = scheduler.add_noise(M_a, distance_map, t)
            pyramid_features = pcn(I, M_v, x_t, t, use_hf=False)
            x_hat_0_logits = dn(x_t, t, pyramid_features)
            
            loss_bce = bce_criterion(x_hat_0_logits, M_a)
            loss_iou = iou_criterion(x_hat_0_logits, M_a)
            loss = loss_bce + loss_iou
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pcn.parameters()) + list(dn.parameters()), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            global_step += 1
            
            if global_step % 10 == 0:
                history['train_steps'].append(global_step)
                history['train_losses'].append(loss.item())
                
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = validate(val_dataloader, scheduler, pcn, dn, criterion)
        lr_scheduler.step()
        
        history['epoch_train_losses'].append(avg_train_loss)
        history['epoch_val_losses'].append(avg_val_loss)
        
        print(f"📊 [{strategy_name.upper()}] Epoch {epoch+1}: Train = {avg_train_loss:.5f} | Val = {avg_val_loss:.5f}")
        
        save_loss_plot_individual(history, indiv_plot_path, strategy_name)
        
        checkpoint_data = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'pcn_state_dict': pcn.state_dict(),
            'dn_state_dict': dn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history 
        }
        
        torch.save(checkpoint_data, latest_ckpt_path)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data['best_val_loss'] = best_val_loss
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"🏆 Cập nhật Best Checkpoint cho [{strategy_name}].")

    # DỌN DẸP RAM GPU TRƯỚC KHI CHUYỂN SANG CHIẾN LƯỢC MỚI
    del pcn, dn, optimizer, scheduler, train_dataloader, val_dataloader
    torch.cuda.empty_cache()
    gc.collect()
    
    return history

EXTERNAL_CHECKPOINTS = {
    "baseline": "/kaggle/input/models/ralphsitinh/cocoa-7epoch/pytorch/default/1/latest_ckpt_baseline.pth",
    "clamped": "/kaggle/input/models/ralphsitinh/cocoa-7epoch/pytorch/default/1/latest_ckpt_clamped.pth",
    "exponential": "/kaggle/input/models/ralphsitinh/cocoa-7epoch/pytorch/default/1/latest_ckpt_exponential.pth"
}

def restore_checkpoints_from_input():
    """
    Hàm tự động quét vùng Input của Kaggle và copy các file tạ (checkpoint)
    vào vùng Working để code có thể đọc và ghi tiếp.
    """
    print("\n" + "="*50)
    print("🔍 HỆ THỐNG KHÔI PHỤC TIẾN TRÌNH (CHECKPOINT RESTORATION)")
    print("="*50)
    
    restored_count = 0
    for strategy, src_path in EXTERNAL_CHECKPOINTS.items():
        dst_path = os.path.join(SAVE_DIR, f"latest_ckpt_{strategy}.pth")
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"✅ Đã nạp thành công tạ của chiến lược [{strategy.upper()}] vào Working Directory.")
            restored_count += 1
        else:
            # Nếu file đã tồn tại sẵn trong Working (do đang train dở trong cùng 1 session) thì không báo lỗi
            if os.path.exists(dst_path):
                print(f"⚡ Tạ của [{strategy.upper()}] đã có sẵn trong Working. Bỏ qua bước copy.")
            else:
                print(f"⚪ Không có tạ nguồn cho [{strategy.upper()}]. Sẽ huấn luyện từ Epoch 1.")
                
    if restored_count > 0:
        print(f"🔥 Đã khôi phục {restored_count}/3 chiến lược. Sẵn sàng chiến đấu tiếp!")

# ==========================================
# TRÌNH QUẢN LÝ TỔNG (MAIN MANAGER)
# ==========================================
def main():
    # 1. Thực hiện copy checkpoint từ Input sang Working (Nếu cờ Resume bật)
    if RESUME_TRAINING:
        restore_checkpoints_from_input()
        
    global_start_time = time.time()
    time_limit_seconds = MAX_TRAIN_HOURS * 3600
    all_histories = {}
    
    # 2. Chạy lần lượt từng chiến lược
    for strategy in STRATEGIES_TO_RUN:
        history = train_strategy(strategy, global_start_time, time_limit_seconds)
        all_histories[strategy] = history
        
        # Cứ xong 1 chiến lược (hoặc bị ngắt), cập nhật ngay biểu đồ tổng hợp
        combined_plot_path = "/kaggle/working/ablation_study_comparison.png"
        save_combined_val_plot(all_histories, combined_plot_path)
        
        # Kiểm tra nếu hết giờ thì dừng vòng lặp lớn
        if (time.time() - global_start_time) > time_limit_seconds:
            print("\n🛑 Hệ thống nhận diện thời gian chạy đã chạm ngưỡng an toàn. Chủ động dừng chương trình.")
            break

    print("\n🏁 TOÀN BỘ QUÁ TRÌNH ABLATION STUDY ĐÃ KẾT THÚC!")
    print("👉 Hãy kiểm tra file 'ablation_study_comparison.png' để xem thành quả!")
    print("💡 Đừng quên download thư mục /kaggle/working/checkpoints về máy tính hoặc save thành Dataset mới nhé!")

if __name__ == "__main__":
    main()