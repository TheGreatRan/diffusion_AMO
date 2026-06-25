import torch
import cv2
import numpy as np

import numpy as np
import cv2
import torch

def get_distance_map(modal_mask: torch.Tensor) -> torch.Tensor:
    """
    Tính ma trận khoảng cách từ biên phần hiển thị (Modal Mask) đến vùng bị che khuất.
    """
    device = modal_mask.device
    masks_np = modal_mask.detach().cpu().numpy()
    batch_size = masks_np.shape[0]
    dist_maps = []

    for i in range(batch_size):
        # Lấy mask của 1 ảnh trong batch
        mask_single = (masks_np[i, 0] > 0.5).astype(np.uint8)
        
        # BẢO VỆ CHỐNG LỖI (EDGE CASE): Vật thể bị che hoàn toàn
        if mask_single.sum() == 0:
            # Gán khoảng cách = 0 toàn ảnh. DA-UST sẽ tự động hoạt động như UST gốc.
            dist_maps.append(np.zeros_like(mask_single, dtype=np.float32))
            continue
            
        inverted_mask = 1 - mask_single
        # Tính khoảng cách Euclidean
        dist = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_maps.append(dist)

    dist_tensor = torch.tensor(np.array(dist_maps), device=device).unsqueeze(1)
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