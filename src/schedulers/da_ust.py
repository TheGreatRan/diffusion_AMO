import torch
import math

class GaussianDAUSTScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.999, beta_end=0.8, sigma=5.0):
        """
        Lịch trình xói mòn lai: Distance-Aware UST + Gaussian Noise.
        """
        self.num_train_timesteps = num_train_timesteps
        self.sigma = sigma
        
        # 1. LỊCH TRÌNH DA-UST (Rời rạc)
        self.ust_betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.ust_alphas_cumprod = torch.cumprod(self.ust_betas, dim=0)

        # 2. LỊCH TRÌNH GAUSSIAN (Liên tục)
        self.gaussian_betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.gaussian_alphas_cumprod = torch.cumprod(1 - self.gaussian_betas, dim=0)

    def add_noise(self, amodal_mask: torch.Tensor, distance_map: torch.Tensor, timesteps: torch.Tensor, return_intermediate=False):
        """
        Thực hiện quy trình Forward hoàn chỉnh: x_0 -> x'_t -> x_t
        
        Args:
            return_intermediate (bool): Nếu True, trả về cả x'_t để debug/hiển thị.
        """
        device = amodal_mask.device
        self.ust_alphas_cumprod = self.ust_alphas_cumprod.to(device)
        self.gaussian_alphas_cumprod = self.gaussian_alphas_cumprod.to(device)

        bar_beta_t = self.ust_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        gauss_alpha_bar_t = self.gaussian_alphas_cumprod[timesteps].view(-1, 1, 1, 1)

        # ==========================================
        # BƯỚC 1: XÓI MÒN CÓ NHẬN THỨC KHOẢNG CÁCH -> x'_t
        # ==========================================
        spatial_decay = torch.exp(-(distance_map ** 2) / (2 * (self.sigma ** 2)))
        p_keep = bar_beta_t * spatial_decay
        
        rand_tensor = torch.rand_like(amodal_mask)
        eroded_mask = (rand_tensor < p_keep).float()
        
        modal_mask = (distance_map == 0).float()
        x_prime_t = torch.where(modal_mask == 1, amodal_mask, eroded_mask * amodal_mask)

        # ==========================================
        # BƯỚC 2: THÊM NHIỄU GAUSSIAN LIÊN TỤC -> x_t
        # ==========================================
        epsilon = torch.randn_like(x_prime_t)
        x_t = torch.sqrt(gauss_alpha_bar_t) * x_prime_t + torch.sqrt(1 - gauss_alpha_bar_t) * epsilon

        if return_intermediate:
            return x_t, x_prime_t
        return x_t

# ==========================================
# KHỐI TEST TRỰC QUAN 3 BƯỚC
# ==========================================
if __name__ == "__main__":
    def print_matrix(tensor, name, fmt="{:4.1f}"):
        print(f"\n--- {name} ---")
        np_arr = tensor[0,0].numpy()
        for row in np_arr:
            print(" ".join([fmt.format(val) for val in row]))

    # 1. Khởi tạo Amodal Mask ban đầu (x_0)
    M_a = torch.zeros((1, 1, 10, 10))
    M_a[0, 0, 2:9, 2:9] = 1.0 # Hình vuông 7x7
    
    # 2. Khởi tạo Distance Map (Tâm d=0, lan dần ra ngoài)
    dist_map = torch.tensor([[
        [4, 3, 2, 2, 2, 2, 3, 4, 5, 6],
        [4, 3, 2, 1, 1, 1, 2, 3, 4, 5],
        [3, 2, 1, 0, 0, 0, 1, 2, 3, 4],
        [3, 2, 1, 0, 0, 0, 1, 2, 3, 4],
        [3, 2, 1, 0, 0, 0, 1, 2, 3, 4],
        [4, 3, 2, 1, 1, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 2, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 3, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 4, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 5, 5, 6, 7, 8, 9]
    ]]).unsqueeze(0).float()
    
    scheduler = GaussianDAUSTScheduler(num_train_timesteps=1000, sigma=3.0)
    t_test = torch.tensor([500]) # Chạy ở bước thời gian 500
    
    # Gọi hàm với cờ return_intermediate=True
    x_t, x_prime_t = scheduler.add_noise(M_a, dist_map, t_test, return_intermediate=True)
    
    # IN KẾT QUẢ ĐỐI CHIẾU
    print_matrix(M_a, "1. TRẠNG THÁI BAN ĐẦU (x_0 - Amodal Mask)", "{:4.0f}")
    print_matrix(x_prime_t, "2. SAU KHI QUA DA-UST (x'_t - Rụng thành 0 ở viền)", "{:4.0f}")
    print_matrix(x_t, f"3. SAU KHI CỘNG GAUSSIAN (x_t - Bước {t_test.item()})", "{:5.2f}")
    
    print("\n✅ KIỂM ĐỊNH TÍNH CHÍNH XÁC:")
    print("- Bước 1 -> Bước 2: Hãy nhìn vào vùng lõi (3x3 ở giữa). Mặc dù chịu xói mòn, nó vẫn giữ vững 100% là số 1 vì d=0 (bọc thép ranh giới). Các vùng rìa xa đã rụng thành số 0.")
    print("- Bước 2 -> Bước 3: Toàn bộ ma trận (cả số 0 và số 1) đều bị xáo trộn nhẹ bởi số thập phân do nhiễu Gaussian liên tục cộng vào.")