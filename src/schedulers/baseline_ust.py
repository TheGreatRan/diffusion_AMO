import torch

class BaselineUSTScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.999, beta_end=0.95):
        """
        Khởi tạo Lịch trình nhiễu Baseline theo đúng bài báo gốc CondDiff-AMO.
        Sử dụng cơ chế lai: Xói mòn đồng đều (UST) + Nhiễu chuẩn (Gaussian).
        
        Args:
            num_train_timesteps (int): Tổng số bước khuếch tán T.
            beta_start (float): Xác suất giữ trạng thái amodal ở t=0.
            beta_end (float): Xác suất giữ trạng thái amodal ở t=T.
        """
        self.num_train_timesteps = num_train_timesteps
        
        # 1. Lịch trình cho UST (Rời rạc - Discrete)
        # beta ở đây là xác suất GIỮ NGUYÊN trạng thái
        self.ust_betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.ust_alphas_cumprod = torch.cumprod(self.ust_betas, dim=0)

        # 2. Lịch trình cho Gaussian Noise (Liên tục - Continuous)
        # Sử dụng DDPM schedule chuẩn (0.0001 đến 0.02)
        self.gaussian_betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.gaussian_alphas_cumprod = torch.cumprod(1 - self.gaussian_betas, dim=0)

    def add_noise(self, amodal_mask: torch.Tensor, modal_mask: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện quy trình Forward 2 bước: x_0 -> x'_t -> x_t
        
        Args:
            amodal_mask (torch.Tensor): Ground truth M_a (B, 1, H, W).
            modal_mask (torch.Tensor): Phần nhìn thấy M_v (B, 1, H, W) dùng để làm mỏ neo cố định.
            timesteps (torch.Tensor): Tensor chứa các bước t ngẫu nhiên (B,).
            
        Returns:
            torch.Tensor: Mask nhiễu lai x_t sẵn sàng đưa vào U-Net.
        """
        device = amodal_mask.device
        self.ust_alphas_cumprod = self.ust_alphas_cumprod.to(device)
        self.gaussian_alphas_cumprod = self.gaussian_alphas_cumprod.to(device)

        # Lấy tham số theo thời gian t cho từng ảnh trong batch
        bar_beta_t = self.ust_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        gauss_alpha_bar_t = self.gaussian_alphas_cumprod[timesteps].view(-1, 1, 1, 1)

        # ==========================================
        # BƯỚC 1: XÓI MÒN RỜI RẠC ĐỒNG ĐỀU (UST) -> x'_t
        # ==========================================
        # Tạo ma trận xác suất ngẫu nhiên U ~ Uniform(0, 1)
        rand_tensor = torch.rand_like(amodal_mask)
        
        # Bất kỳ pixel nào có U < bar_beta_t sẽ giữ được màu trắng. 
        # (LƯU Ý: Baseline cào bằng xác suất này trên toàn ảnh, không dùng khoảng cách d)
        eroded_mask = (rand_tensor < bar_beta_t).float()
        
        # Ép đè phần Modal Mask: M_v tuyệt đối không bị xói mòn
        x_prime_t = torch.where(modal_mask == 1, amodal_mask, eroded_mask * amodal_mask)

        # ==========================================
        # BƯỚC 2: NHIỄU GAUSSIAN LIÊN TỤC -> x_t
        # ==========================================
        # Tạo nhiễu trắng epsilon ~ N(0, 1)
        epsilon = torch.randn_like(x_prime_t)
        
        # Trộn theo công thức DDPM kinh điển: sqrt(alpha_bar) * x + sqrt(1 - alpha_bar) * epsilon
        x_t = torch.sqrt(gauss_alpha_bar_t) * x_prime_t + torch.sqrt(1 - gauss_alpha_bar_t) * epsilon

        return x_t

# ==========================================
# KHỐI TEST CỤC BỘ
# ==========================================
if __name__ == "__main__":
    # Giả lập: Ảnh 10x10, Amodal (2 đến 8), Modal (2 đến 4)
    amodal_mask = torch.zeros((1, 1, 10, 10))
    amodal_mask[0, 0, 2:9, 2:9] = 1.0
    
    modal_mask = torch.zeros((1, 1, 10, 10))
    modal_mask[0, 0, 2:5, 2:5] = 1.0
    
    scheduler = BaselineUSTScheduler(num_train_timesteps=1000)
    
    # Test với t = 500 (Giữa quá trình)
    t_test = torch.tensor([500])
    x_t = scheduler.add_noise(amodal_mask, modal_mask, t_test)
    
    print("--- Amodal Mask Gốc (x_0) ---")
    print(amodal_mask[0,0].int())
    
    print(f"\n--- Nhiễu Lai x_t ở t={t_test.item()} ---")
    # Hiển thị x_t với 1 chữ số thập phân để thấy rõ nhiễu Gaussian liên tục
    np_xt = x_t[0,0].numpy()
    for row in np_xt:
        print(" ".join([f"{val:4.1f}" for val in row]))
    
    print("\nNhận xét: Bạn sẽ thấy phần lõi và phần viền bị xói mòn NGẪU NHIÊN như nhau (không có mỏ neo viền).")
    print("Và các giá trị không còn là 0, 1 tròn trĩnh nữa mà là số thập phân do đã cộng thêm Gaussian!")