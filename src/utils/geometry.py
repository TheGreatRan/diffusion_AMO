import torch
import cv2
import numpy as np

def get_distance_map(modal_mask: torch.Tensor) -> torch.Tensor:
    """
    Tính toán ma trận khoảng cách Euclidean từ vùng bị che khuất đến ranh giới Modal Mask.
    Hỗ trợ xử lý theo Batch và tự động đồng bộ thiết bị (CPU/GPU).
    
    Args:
        modal_mask (torch.Tensor): Tensor nhị phân có kích thước (B, 1, H, W) hoặc (B, H, W).
                                   Giá trị 1: Vùng nhìn thấy (Modal), Giá trị 0: Vùng che khuất (Nền).
                                   
    Returns:
        torch.Tensor: Tensor chứa khoảng cách d_{i,j}, cùng kích thước và device với modal_mask.
    """
    # 1. Lưu lại device (cuda/cpu) và kích thước gốc để trả về sau
    original_device = modal_mask.device
    original_shape = modal_mask.shape
    
    # 2. Xử lý chiều dữ liệu để đảm bảo vòng lặp chạy đúng (đưa về B, H, W)
    if len(original_shape) == 4: # Dạng (Batch, Channel, Height, Width)
        masks_np = modal_mask.squeeze(1).detach().cpu().numpy()
    elif len(original_shape) == 3: # Dạng (Batch, Height, Width)
        masks_np = modal_mask.detach().cpu().numpy()
    else:
        raise ValueError(f"modal_mask phải có 3 hoặc 4 chiều, nhưng nhận được {len(original_shape)}")
        
    batch_size = masks_np.shape[0]
    dist_maps = []
    
    # 3. Chạy vòng lặp tính toán khoảng cách cho từng ảnh trong Batch
    for i in range(batch_size):
        # Ép kiểu về nhị phân cứng (chỉ có 0 và 1) với định dạng uint8 cho OpenCV
        mask_single = (masks_np[i] > 0.5).astype(np.uint8)
        
        # [QUAN TRỌNG]: Thủ thuật đảo ngược (Invert)
        # OpenCV tính khoảng cách từ các điểm (1) đến điểm (0) gần nhất.
        # Chúng ta muốn tính khoảng cách tới M_v (đang có giá trị 1).
        # Nên ta phải đảo ngược: M_v thành 0 (đích đến), vùng bị che khuất thành 1 (điểm xuất phát).
        inverted_mask = 1 - mask_single
        
        # Tính khoảng cách Euclidean L2 chính xác cao
        dist = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        dist_maps.append(dist)
        
    # 4. Gộp (Stack) list các numpy array về lại PyTorch Tensor và đưa về lại GPU (nếu có)
    dist_tensor = torch.tensor(np.stack(dist_maps), dtype=torch.float32, device=original_device)
    
    # 5. Khôi phục lại số chiều Channel gốc nếu cần
    if len(original_shape) == 4:
        dist_tensor = dist_tensor.unsqueeze(1) # Trở về dạng (B, 1, H, W)
        
    return dist_tensor

# Đoạn mã Test cục bộ (Chỉ chạy khi bạn thực thi trực tiếp file này)
if __name__ == "__main__":
    # Giả lập một batch gồm 2 ảnh, 1 kênh, kích thước 5x5
    dummy_batch = torch.zeros((2, 1, 5, 5)).cuda() if torch.cuda.is_available() else torch.zeros((2, 1, 5, 5))
    
    # Vẽ một vật thể (Modal Mask) ở góc trên bên trái của ảnh đầu tiên
    dummy_batch[0, 0, 0:2, 0:2] = 1.0 
    
    print("--- Modal Mask Đầu vào (Ảnh 1) ---")
    print(dummy_batch[0, 0])
    
    print("\n--- Ma trận Khoảng cách Đầu ra (Ảnh 1) ---")
    distances = get_distance_map(dummy_batch)
    print(distances[0, 0])