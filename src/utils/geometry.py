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

def compute_occlusion_weight_map(
    amodal_mask: torch.Tensor,
    modal_mask: torch.Tensor,
    occlusion_weight: float = 3.0,
    background_weight: float = 1.0,
) -> torch.Tensor:
    """
    Tính weight_map để đưa vào WeightedBCELoss / WeightedIoULoss, ƯU TIÊN vùng bị che khuất
    (occluded region = phần thuộc amodal mask nhưng KHÔNG thuộc modal/visible mask).

    Lý do cần hàm này: nếu loss cộng đều trên toàn bộ pixel, vùng visible (thường chiếm đa số
    diện tích, và gần như "cho sẵn" trong M_v) sẽ áp đảo gradient, khiến model không có động lực
    học tốt vùng occluded (khó, đòi hỏi suy luận ngữ nghĩa từ ảnh) -> mIoU_inv thấp bất thường
    dù mIoU_full vẫn ổn (model chỉ đang "chép" gần đúng M_v).

    Args:
        amodal_mask (torch.Tensor): Ground truth M_a, shape (B, 1, H, W), giá trị {0, 1}.
        modal_mask (torch.Tensor): Modal/visible mask M_v, shape (B, 1, H, W), giá trị {0, 1}.
        occlusion_weight (float): Trọng số áp cho vùng occluded (M_a=1, M_v=0). Mặc định 3.0
            nghĩa là lỗi ở vùng khuất bị phạt nặng gấp 3 lần lỗi ở vùng còn lại.
        background_weight (float): Trọng số cho phần còn lại (visible + background). Mặc định 1.0.

    Returns:
        torch.Tensor: weight_map cùng shape (B, 1, H, W), giá trị trong {background_weight, occlusion_weight}.
    """
    # Vùng occluded = thuộc amodal NHƯNG không thuộc modal (bị che, cần model tự suy luận)
    occluded_region = amodal_mask * (1.0 - modal_mask)

    weight_map = background_weight + (occlusion_weight - background_weight) * occluded_region
    return weight_map

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