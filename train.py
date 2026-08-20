"""
train.py — CondDiff-AMO (AAAI-26 reproduction), one-stage training.

Chuyển thể từ notebook `amodal_fix.ipynb` để chạy được bằng CLI (không phụ thuộc Kaggle cell),
đã tích hợp sẵn fix weight_map ưu tiên vùng occluded (xem src/utils/geometry.py ->
compute_occlusion_weight_map).

Cách chạy ví dụ:
    python train.py --mode supervised --strategy baseline
    python train.py --mode zero-shot --strategy clamped --occlusion-weight 3.0
    python train.py --mode supervised --strategy exponential --no-ocr   # ablation không OCR

Chạy `python train.py --help` để xem đầy đủ tham số.
"""

import argparse
import gc
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")  # Không cần màn hình (chạy trên server/Kaggle)
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from src.dataset import AmodalDataset
from src.models.denoising_net import (
    BoundaryAwareEdgeLoss,
    DenoisingNetwork,
    WeightedBCELoss,
    WeightedIoULoss,
)
from src.models.pcn_extractor import PCNExtractor
from src.schedulers.da_ust import (
    BaselineUSTScheduler,
    ClampedDAUSTScheduler,
    ExponentialDAUSTScheduler,
)
from src.utils.geometry import compute_occlusion_weight_map, get_distance_map


# ============================================================
# ⚙️  CONFIG — giá trị mặc định khớp CONFIG cell của notebook.
# Có thể override bằng CLI args (xem parse_args() ở cuối file).
# ============================================================
EXPERIMENT_MODE = "zero-shot"     # "zero-shot" (train Pix2gestalt) | "supervised" (train COCOA)
STRATEGY_NAME = "clamped"         # "baseline" | "clamped" | "exponential"

NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)
TIMESTEPS = 1000
MAX_TRAIN_HOURS = 11.5

# FIX: trọng số ưu tiên vùng occluded trong BCE/IoU loss (trước đây luôn = None -> uniform).
OCCLUSION_LOSS_WEIGHT = 3.0

USE_OCR = True
PRETRAINED_BACKBONE = True
MODEL_NAME = "pvt_v2_b4"

COCO_IMG_DIR = "/kaggle/input/datasets/ralphsitinh/cocoa-image/cocoa_images_extracted"
PIX2GESTALT_DIR = "/kaggle/input/datasets/ralphsitinh/data-pix2geltat/pix2gestalt_occlusions_release"
COCOA_TRAIN_JSON = "/kaggle/input/datasets/ralphsitinh/coco-amodal-annotations/annotations/COCO_amodal_train2014.json"
COCOA_VAL_JSON = "/kaggle/input/datasets/ralphsitinh/coco-amodal-annotations/annotations/COCO_amodal_val2014.json"

SAVE_DIR = "/kaggle/working/checkpoints"
RESUME_FROM_EXTERNAL = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Các biến dẫn xuất (được tính lại trong configure_derived())
DATA_DIR_TARGET = None
MODE_TARGET = None
LATEST_CKPT_PATH = None
BEST_CKPT_PATH = None
INDIV_PLOT_PATH = None


def configure_derived():
    """Tính lại các biến phụ thuộc CONFIG (gọi lại sau khi override qua CLI)."""
    global DATA_DIR_TARGET, MODE_TARGET, LATEST_CKPT_PATH, BEST_CKPT_PATH, INDIV_PLOT_PATH

    os.makedirs(SAVE_DIR, exist_ok=True)

    if EXPERIMENT_MODE == "zero-shot":
        DATA_DIR_TARGET = PIX2GESTALT_DIR
        MODE_TARGET = "pix2gestalt"
    elif EXPERIMENT_MODE == "supervised":
        DATA_DIR_TARGET = COCOA_TRAIN_JSON
        MODE_TARGET = "cocoa_train"
    else:
        raise ValueError(f"EXPERIMENT_MODE không hợp lệ: {EXPERIMENT_MODE}")

    LATEST_CKPT_PATH = os.path.join(SAVE_DIR, f"latest_ckpt_{STRATEGY_NAME}_{EXPERIMENT_MODE}.pth")
    BEST_CKPT_PATH = os.path.join(SAVE_DIR, f"best_ckpt_{STRATEGY_NAME}_{EXPERIMENT_MODE}.pth")
    INDIV_PLOT_PATH = f"/kaggle/working/loss_chart_{STRATEGY_NAME}_{EXPERIMENT_MODE}.png"

    print("🎯 Cấu hình One-stage Training:")
    print(f"   Mode                 : {EXPERIMENT_MODE}")
    print(f"   Strategy             : {STRATEGY_NAME}")
    print(f"   Tổng epochs          : {NUM_EPOCHS}")
    print(f"   Learning rate        : {LEARNING_RATE}")
    print(f"   Occlusion loss weight: {OCCLUSION_LOSS_WEIGHT}")
    print(f"   use_ocr              : {USE_OCR}")
    print(f"   Backbone             : {MODEL_NAME} (pretrained={PRETRAINED_BACKBONE})")
    print(f"   Data dir             : {DATA_DIR_TARGET}")
    print(f"   Device               : {DEVICE}")


# ============================================================
# TIỆN ÍCH: VẼ BIỂU ĐỒ LOSS
# ============================================================
def save_loss_plot_individual(history, save_path, strategy_name, phase_name):
    """Vẽ biểu đồ loss cho phase/strategy hiện tại."""
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_steps"], history["train_losses"], label="Train Loss", alpha=0.6, color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"[{strategy_name.upper()} | {phase_name.upper()}] Train Loss (Steps)")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    if len(history["epoch_train_losses"]) > 0:
        epochs = range(1, len(history["epoch_train_losses"]) + 1)
        plt.plot(epochs, history["epoch_train_losses"], label="Train", marker="o")
        plt.plot(epochs, history["epoch_val_losses"], label="Val", marker="s")
        plt.xlabel("Epochs")
        plt.title(f"[{strategy_name.upper()} | {phase_name.upper()}] Train vs Val Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 Đã lưu biểu đồ tại: {save_path}")


# ============================================================
# SAMPLER: CHO PHÉP RESUME GIỮA CHỪNG 1 EPOCH
# ============================================================
class ResumableSampler(Sampler):
    """
    Sampler cho phép resume TRONG một epoch (không chỉ ở ranh giới epoch).
    - Mỗi epoch có một permutation cố định (seed + epoch).
    - start_index cho biết vị trí (theo sample) cần bỏ qua khi resume dở dang.
    """

    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch, start_index=0):
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(indices[self.start_index:])

    def __len__(self):
        return len(self.data_source) - self.start_index


# ============================================================
# KHỞI TẠO TOÀN BỘ HỆ THỐNG (data, model, loss, optimizer, scheduler)
# ============================================================
def build_system():
    print("\n" + "=" * 50)
    print(f"🚀 KHỞI TẠO HỆ THỐNG: [{STRATEGY_NAME.upper()}] — MODE [{EXPERIMENT_MODE.upper()}]")
    print("=" * 50)

    train_dataset = AmodalDataset(
        data_dir=DATA_DIR_TARGET,
        mode=MODE_TARGET,
        image_size=IMAGE_SIZE,
        coco_img_dir=COCO_IMG_DIR,
    )
    sampler = ResumableSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = AmodalDataset(
        data_dir=COCOA_VAL_JSON, mode="cocoa_val", image_size=IMAGE_SIZE, coco_img_dir=COCO_IMG_DIR
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, pin_memory=True
    )

    if STRATEGY_NAME == "baseline":
        scheduler = BaselineUSTScheduler(num_train_timesteps=TIMESTEPS)
    elif STRATEGY_NAME == "clamped":
        scheduler = ClampedDAUSTScheduler(num_train_timesteps=TIMESTEPS, sigma=5.0, beta_min=0.05)
    elif STRATEGY_NAME == "exponential":
        scheduler = ExponentialDAUSTScheduler(num_train_timesteps=TIMESTEPS, gamma=5.0, beta_min=0.05)
    else:
        raise ValueError(f"Không nhận diện được chiến lược: {STRATEGY_NAME}")

    pcn = PCNExtractor(model_name=MODEL_NAME, pretrained=PRETRAINED_BACKBONE).to(DEVICE)
    dn = DenoisingNetwork(pcn_channels=pcn.feature_channels, fuse_channels=256, use_ocr=USE_OCR).to(DEVICE)

    bce_criterion = WeightedBCELoss().to(DEVICE)
    iou_criterion = WeightedIoULoss().to(DEVICE)
    edge_criterion = BoundaryAwareEdgeLoss(weight_boundary=2.0).to(DEVICE)

    optimizer = optim.AdamW(
        [
            {"params": pcn.parameters(), "lr": LEARNING_RATE * 0.1},
            {"params": dn.parameters(), "lr": LEARNING_RATE},
        ],
        weight_decay=1e-2,
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    return (
        train_dataloader,
        val_dataloader,
        sampler,
        scheduler,
        pcn,
        dn,
        bce_criterion,
        iou_criterion,
        edge_criterion,
        optimizer,
        lr_scheduler,
    )


# ============================================================
# VALIDATION LOOP
# ============================================================
def validate(val_dataloader, scheduler, pcn, dn, bce_criterion, iou_criterion, edge_criterion):
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
            # FIX: weight_map ưu tiên vùng occluded (trước đây không truyền -> uniform loss)
            weight_map = compute_occlusion_weight_map(M_a, M_v, occlusion_weight=OCCLUSION_LOSS_WEIGHT)

            pyramid_features = pcn(I, M_v, x_t, t, use_hf=False)
            x_hat_0_logits = dn(x_t, t, pyramid_features)

            loss_bce = bce_criterion(x_hat_0_logits, M_a, weight_map)
            loss_iou = iou_criterion(x_hat_0_logits, M_a, weight_map)
            loss_edge = edge_criterion(x_hat_0_logits, M_a, M_v)
            loss = loss_bce + loss_iou + 0.5 * loss_edge
            val_loss += loss.item()

    return val_loss / len(val_dataloader)


# ============================================================
# KHÔI PHỤC CHECKPOINT TỪ NGUỒN NGOÀI (VD: /kaggle/input/... của session trước)
# ============================================================
def restore_external_checkpoint():
    """
    Nếu người dùng chỉ định RESUME_FROM_EXTERNAL (checkpoint từ lần chạy trước,
    ví dụ nằm ở /kaggle/input/...), copy nó vào SAVE_DIR để logic resume bình thường nhận diện được.
    """
    if RESUME_FROM_EXTERNAL is None:
        return

    if not os.path.exists(RESUME_FROM_EXTERNAL):
        print(f"⚠️ CẢNH BÁO: Không tìm thấy file RESUME_FROM_EXTERNAL tại: {RESUME_FROM_EXTERNAL}")
        print("   Kiểm tra lại đường dẫn Input Dataset đã được Add Data đúng chưa.")
        return

    if os.path.exists(LATEST_CKPT_PATH):
        print("⚡ Đã có checkpoint sẵn trong Working (cùng session). Bỏ qua copy từ external.")
        return

    shutil.copy(RESUME_FROM_EXTERNAL, LATEST_CKPT_PATH)
    print(f"✅ Đã copy checkpoint từ lần chạy trước vào: {LATEST_CKPT_PATH}")


# ============================================================
# VÒNG LẶP TRAIN CHÍNH (có checkpoint/resume, giới hạn thời gian phiên)
# ============================================================
def train_model(global_start_time, time_limit_seconds):
    (
        train_dataloader,
        val_dataloader,
        sampler,
        scheduler,
        pcn,
        dn,
        bce_criterion,
        iou_criterion,
        edge_criterion,
        optimizer,
        lr_scheduler,
    ) = build_system()

    start_epoch = 0
    resume_sample_idx = 0
    best_val_loss = float("inf")
    global_step = 0
    history = {"train_steps": [], "train_losses": [], "epoch_train_losses": [], "epoch_val_losses": []}

    restore_external_checkpoint()

    if os.path.exists(LATEST_CKPT_PATH):
        print(f"♻️ Tìm thấy checkpoint của [{STRATEGY_NAME} | {EXPERIMENT_MODE}]. Đang khôi phục...")
        checkpoint = torch.load(LATEST_CKPT_PATH, map_location=DEVICE)
        pcn.load_state_dict(checkpoint["pcn_state_dict"])
        dn.load_state_dict(checkpoint["dn_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        resume_sample_idx = checkpoint.get("resume_sample_idx", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", history)
        global_step = checkpoint.get("global_step", 0)
        print(f"✅ Khôi phục: Epoch {start_epoch + 1}, sample offset {resume_sample_idx}, global_step {global_step}")
    else:
        print(f"🆕 Không tìm thấy checkpoint cũ. Bắt đầu [{STRATEGY_NAME} | {EXPERIMENT_MODE}] từ đầu.")

    if start_epoch >= NUM_EPOCHS:
        print(f"🎉 [{STRATEGY_NAME} | {EXPERIMENT_MODE}] đã hoàn thành đủ {NUM_EPOCHS} epochs.")
        return history

    for epoch in range(start_epoch, NUM_EPOCHS):
        if (time.time() - global_start_time) > time_limit_seconds:
            print(f"\n⏰ Hết giờ! Tạm ngắt [{STRATEGY_NAME} | {EXPERIMENT_MODE}]...")
            return history

        offset = resume_sample_idx if epoch == start_epoch else 0
        sampler.set_epoch(epoch, start_index=offset)

        pcn.train()
        dn.train()
        train_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"[{STRATEGY_NAME.upper()}|{EXPERIMENT_MODE.upper()}] Ep {epoch+1}/{NUM_EPOCHS}")
        time_up = False

        for batch_idx, batch in enumerate(pbar):
            I, M_v, M_a = [x.to(DEVICE) for x in batch]

            optimizer.zero_grad()
            batch_size = I.shape[0]
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
            distance_map = get_distance_map(M_v)
            # FIX: weight_map ưu tiên vùng occluded (trước đây không truyền -> uniform loss)
            weight_map = compute_occlusion_weight_map(M_a, M_v, occlusion_weight=OCCLUSION_LOSS_WEIGHT)

            x_t = scheduler.add_noise(M_a, distance_map, t)
            pyramid_features = pcn(I, M_v, x_t, t, use_hf=False)
            x_hat_0_logits = dn(x_t, t, pyramid_features)

            loss_bce = bce_criterion(x_hat_0_logits, M_a, weight_map)
            loss_iou = iou_criterion(x_hat_0_logits, M_a, weight_map)
            loss_edge = edge_criterion(x_hat_0_logits, M_a, M_v)
            loss = loss_bce + loss_iou + 0.5 * loss_edge

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pcn.parameters()) + list(dn.parameters()), 1.0)
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                history["train_steps"].append(global_step)
                history["train_losses"].append(loss.item())

            time_up = (time.time() - global_start_time) > time_limit_seconds - 300

            if global_step % 500 == 0 or time_up:
                resume_sample_idx = offset + (batch_idx + 1) * BATCH_SIZE
                checkpoint_data = {
                    "mode": EXPERIMENT_MODE,
                    "epoch": epoch,
                    "resume_sample_idx": resume_sample_idx,
                    "global_step": global_step,
                    "pcn_state_dict": pcn.state_dict(),
                    "dn_state_dict": dn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                }
                torch.save(checkpoint_data, LATEST_CKPT_PATH)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if time_up:
                print(f"\n⏰ Hết giờ giữa epoch! Đã lưu checkpoint tại sample {resume_sample_idx}.")
                return history

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = validate(val_dataloader, scheduler, pcn, dn, bce_criterion, iou_criterion, edge_criterion)
        lr_scheduler.step()

        history["epoch_train_losses"].append(avg_train_loss)
        history["epoch_val_losses"].append(avg_val_loss)

        print(
            f"📊 [{STRATEGY_NAME.upper()}|{EXPERIMENT_MODE.upper()}] Epoch {epoch+1}: "
            f"Train = {avg_train_loss:.5f} | Val = {avg_val_loss:.5f}"
        )

        save_loss_plot_individual(history, INDIV_PLOT_PATH, STRATEGY_NAME, EXPERIMENT_MODE)

        checkpoint_data = {
            "mode": EXPERIMENT_MODE,
            "epoch": epoch + 1,
            "resume_sample_idx": 0,
            "global_step": global_step,
            "pcn_state_dict": pcn.state_dict(),
            "dn_state_dict": dn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }
        torch.save(checkpoint_data, LATEST_CKPT_PATH)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data["best_val_loss"] = best_val_loss
            torch.save(checkpoint_data, BEST_CKPT_PATH)
            print(f"🏆 Cập nhật Best Checkpoint cho [{STRATEGY_NAME} | {EXPERIMENT_MODE}].")

    print(f"\n🎉 [{STRATEGY_NAME} | {EXPERIMENT_MODE}] đã hoàn thành đủ {NUM_EPOCHS} epochs!")

    del pcn, dn, optimizer, scheduler, train_dataloader, val_dataloader
    torch.cuda.empty_cache()
    gc.collect()

    return history


# ============================================================
# CLI ARGPARSE — override CONFIG mặc định
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train CondDiff-AMO (one-stage, AAAI-26 reproduction)")

    parser.add_argument("--mode", dest="mode", choices=["zero-shot", "supervised"], default=EXPERIMENT_MODE)
    parser.add_argument("--strategy", dest="strategy", choices=["baseline", "clamped", "exponential"], default=STRATEGY_NAME)

    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--max-hours", type=float, default=MAX_TRAIN_HOURS)

    parser.add_argument("--occlusion-weight", type=float, default=OCCLUSION_LOSS_WEIGHT,
                         help="Trọng số ưu tiên vùng occluded trong BCE/IoU loss. =1.0 để tắt (uniform, hành vi cũ).")

    parser.add_argument("--use-ocr", dest="use_ocr", action="store_true", default=USE_OCR)
    parser.add_argument("--no-ocr", dest="use_ocr", action="store_false",
                         help="Tắt module OCR trong DenoisingNetwork (dùng cho ablation, Table 4).")

    parser.add_argument("--model-name", default=MODEL_NAME, help="Backbone PVTv2 (vd: pvt_v2_b4).")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=PRETRAINED_BACKBONE,
                         help="Không tải weight ImageNet pretrained cho backbone.")

    parser.add_argument("--cocoa-img-dir", default=COCO_IMG_DIR)
    parser.add_argument("--pix2gestalt-dir", default=PIX2GESTALT_DIR)
    parser.add_argument("--cocoa-train-json", default=COCOA_TRAIN_JSON)
    parser.add_argument("--cocoa-val-json", default=COCOA_VAL_JSON)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--resume-from-external", default=RESUME_FROM_EXTERNAL)

    return parser.parse_args()


def apply_args(args):
    """Ghi đè CONFIG mặc định bằng giá trị từ CLI, rồi tính lại các biến dẫn xuất."""
    global EXPERIMENT_MODE, STRATEGY_NAME, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, IMAGE_SIZE
    global TIMESTEPS, MAX_TRAIN_HOURS, OCCLUSION_LOSS_WEIGHT, USE_OCR, MODEL_NAME, PRETRAINED_BACKBONE
    global COCO_IMG_DIR, PIX2GESTALT_DIR, COCOA_TRAIN_JSON, COCOA_VAL_JSON, SAVE_DIR, RESUME_FROM_EXTERNAL

    EXPERIMENT_MODE = args.mode
    STRATEGY_NAME = args.strategy
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = (args.image_size, args.image_size)
    TIMESTEPS = args.timesteps
    MAX_TRAIN_HOURS = args.max_hours
    OCCLUSION_LOSS_WEIGHT = args.occlusion_weight
    USE_OCR = args.use_ocr
    MODEL_NAME = args.model_name
    PRETRAINED_BACKBONE = args.pretrained
    COCO_IMG_DIR = args.cocoa_img_dir
    PIX2GESTALT_DIR = args.pix2gestalt_dir
    COCOA_TRAIN_JSON = args.cocoa_train_json
    COCOA_VAL_JSON = args.cocoa_val_json
    SAVE_DIR = args.save_dir
    RESUME_FROM_EXTERNAL = args.resume_from_external

    configure_derived()


def main():
    args = parse_args()
    apply_args(args)

    global_start_time = time.time()
    time_limit_seconds = MAX_TRAIN_HOURS * 3600

    history = train_model(global_start_time, time_limit_seconds)

    print("\n🏁 KẾT THÚC LẦN CHẠY NÀY.")
    print(f"👉 Checkpoint mới nhất: {LATEST_CKPT_PATH}")
    print(f"👉 Checkpoint tốt nhất: {BEST_CKPT_PATH}")

    return history


if __name__ == "__main__":
    main()