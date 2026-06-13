import torch

class AverageMeter(object):
    """
    Lớp tiện ích kinh điển trong PyTorch để theo dõi và tính toán 
    giá trị trung bình của Loss hoặc Metric qua từng Batch.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_amodal_metrics(pred_mask: torch.Tensor, gt_amodal: torch.Tensor, gt_modal: torch.Tensor, threshold: float = 0.5):
    """
    Tính toán các chỉ số mIoU theo chuẩn bài báo CondDiff-AMO.
    
    Args:
        pred_mask (Tensor): Dự đoán của mạng DN (sau khi qua Sigmoid), shape (B, 1, H, W)
        gt_amodal (Tensor): Ground truth Amodal mask (0 hoặc 1), shape (B, 1, H, W)
        gt_modal (Tensor): Ground truth Modal mask (Phần hiển thị), shape (B, 1, H, W)
        threshold (float): Ngưỡng để nhị phân hóa xác suất thành pixel trắng/đen.
        
    Returns:
        iou_batch (Tensor): IoU tổng thể cho từng ảnh trong batch.
        inv_iou_batch (Tensor): Invisible IoU cho từng ảnh.
        valid_inv_mask (Tensor): Mảng boolean đánh dấu xem ảnh đó có thực sự bị che khuất hay không.
    """
    # 1. Nhị phân hóa ma trận dự đoán
    pred = (pred_mask > threshold).to(torch.int64)
    gt_a = gt_amodal.to(torch.int64)
    gt_m = gt_modal.to(torch.int64)
    
    # ==========================================
    # 2. TÍNH AMODAL IoU (Toàn bộ vật thể)
    # ==========================================
    intersection = ((pred == 1) & (gt_a == 1)).sum(dim=(1, 2, 3))
    union = ((pred == 1) | (gt_a == 1)).sum(dim=(1, 2, 3))
    
    iou_batch = intersection.float() / (union.float() + 1e-6)
    
    # ==========================================
    # 3. TÍNH INVISIBLE IoU (Chỉ vùng bị che khuất)
    # ==========================================
    # Vùng bị che (Invisible) là nơi: Ground Truth Amodal là 1 NHƯNG Modal là 0
    # Ta dùng phép toán logic mask: (gt_m == 0) để chỉ tập trung vào vùng này
    inv_intersection = ((pred == 1) & (gt_a == 1) & (gt_m == 0)).sum(dim=(1, 2, 3))
    inv_union = (((pred == 1) | (gt_a == 1)) & (gt_m == 0)).sum(dim=(1, 2, 3))
    
    inv_iou_batch = inv_intersection.float() / (inv_union.float() + 1e-6)
    
    # Xử lý trường hợp ngoại lệ: Nếu một vật thể hoàn toàn KHÔNG bị che (Modal = Amodal)
    # thì inv_union sẽ bằng 0. Ta không được tính ảnh này vào trung bình của Invisible IoU.
    valid_inv_mask = inv_union > 0
    
    return iou_batch, inv_iou_batch, valid_inv_mask

# ==========================================
# KHỐI TEST CỤC BỘ
# ==========================================
if __name__ == "__main__":
    # Giả lập 1 batch gồm 2 ảnh kích thước 10x10
    pred = torch.zeros((2, 1, 10, 10))
    gt_a = torch.zeros((2, 1, 10, 10))
    gt_m = torch.zeros((2, 1, 10, 10))
    
    # Ảnh 1: Vật thể bị che mất 1 nửa
    gt_a[0, 0, 2:8, 2:8] = 1.0 # Hình vuông 6x6 (Diện tích 36)
    gt_m[0, 0, 2:8, 2:5] = 1.0 # Chỉ nhìn thấy nửa bên trái (Diện tích 18)
    pred[0, 0, 2:8, 2:7] = 0.8 # Mô hình dự đoán đúng nửa trái, nhưng hụt một chút ở nửa phải
    
    # Ảnh 2: Vật thể KHÔNG bị che (Để test valid_inv_mask)
    gt_a[1, 0, 4:6, 4:6] = 1.0
    gt_m[1, 0, 4:6, 4:6] = 1.0
    pred[1, 0, 4:6, 4:6] = 0.9

    iou, inv_iou, valid_mask = compute_amodal_metrics(pred, gt_a, gt_m)
    
    print("--- KẾT QUẢ ĐÁNH GIÁ (METRICS) ---")
    print(f"Ảnh 1 - Vật thể bị che:")
    print(f"  + Amodal IoU   : {iou[0].item():.4f} (Mong đợi: < 1.0 vì dự đoán hụt)")
    print(f"  + Invisible IoU: {inv_iou[0].item():.4f} (Chỉ tính độ chính xác của nửa bên phải)")
    print(f"  + Có tính vào mIoU Invisible không?: {valid_mask[0].item()}")
    
    print(f"\nẢnh 2 - Vật thể không bị che:")
    print(f"  + Amodal IoU   : {iou[1].item():.4f} (Mong đợi: 1.0 vì dự đoán hoàn hảo)")
    print(f"  + Có tính vào mIoU Invisible không?: {valid_mask[1].item()} (Vì không có phần che khuất)")