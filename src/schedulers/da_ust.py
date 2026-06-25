import torch
import math

class BaseDAUSTScheduler:
    def __init__(self, num_train_timesteps=1000, sigma=5.0, beta_min=0.05, gamma=5.0, image_size=256):
        self.num_train_timesteps = num_train_timesteps
        self.sigma = sigma
        self.beta_min = beta_min
        self.gamma = gamma
        self.image_size = image_size
        
        # Hằng số chuẩn hóa toàn cục (Đường chéo ảnh)
        self.d_global = math.sqrt(image_size**2 + image_size**2)
        
        # ==========================================
        # FIXED: COSINE SCHEDULE CHO ABSORBING STATE
        # ==========================================
        # Định nghĩa trực tiếp bar_beta_t (xác suất giữ lại) bằng đường cong Cosine
        # Giúp mượt mà, chống sụp đổ ở các bước giữa (t=500 vẫn còn ~50% mask)
        steps = torch.arange(num_train_timesteps + 1, dtype=torch.float32) / num_train_timesteps
        s = 0.008 # Tham số dịch chuyển nhẹ
        alphas = torch.cos(((steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        
        # ust_alphas_cumprod chính là bar_beta_t
        self.ust_alphas_cumprod = alphas / alphas[0] 
        # Cắt bỏ phần tử đầu tiên để mảng có đúng num_train_timesteps phần tử
        self.ust_alphas_cumprod = self.ust_alphas_cumprod[1:]

        # Lịch trình Gaussian (Vẫn giữ nguyên tuyến tính chuẩn của DDPM)
        self.gaussian_betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.gaussian_alphas_cumprod = torch.cumprod(1 - self.gaussian_betas, dim=0)

    def _compute_p_keep(self, bar_beta_t: torch.Tensor, distance_map: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Bạn phải định nghĩa logic tính p_keep ở class con!")

    def add_noise(self, amodal_mask: torch.Tensor, distance_map: torch.Tensor, timesteps: torch.Tensor, return_intermediate=False):
        device = amodal_mask.device
        self.ust_alphas_cumprod = self.ust_alphas_cumprod.to(device)
        self.gaussian_alphas_cumprod = self.gaussian_alphas_cumprod.to(device)

        # Lấy ra bar_beta_t tương ứng với từng t trong batch
        bar_beta_t = self.ust_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        gauss_alpha_bar_t = self.gaussian_alphas_cumprod[timesteps].view(-1, 1, 1, 1)

        # 1. TÍNH TOÁN XÁC SUẤT GIỮ LẠI
        p_keep = self._compute_p_keep(bar_beta_t, distance_map)
        
        # 2. XÓI MÒN RỜI RẠC -> x'_t
        rand_tensor = torch.rand_like(amodal_mask)
        eroded_mask = (rand_tensor < p_keep).float()
        
        modal_mask = (distance_map == 0).float()
        x_prime_t = torch.where(modal_mask == 1, amodal_mask, eroded_mask * amodal_mask)

        # 3. THÊM NHIỄU GAUSSIAN LIÊN TỤC -> x_t
        epsilon = torch.randn_like(x_prime_t)
        x_t = torch.sqrt(gauss_alpha_bar_t) * x_prime_t + torch.sqrt(1 - gauss_alpha_bar_t) * epsilon

        if return_intermediate:
            return x_t, x_prime_t
        return x_t

# ==========================================
# CÁC CHIẾN LƯỢC TOÁN HỌC (ĐÃ LỌC BỎ CÁC BẢN LỖI)
# ==========================================

class BaselineUSTScheduler(BaseDAUSTScheduler):
    def _compute_p_keep(self, bar_beta_t, distance_map):
        return bar_beta_t.expand_as(distance_map)

class ClampedDAUSTScheduler(BaseDAUSTScheduler):
    def _compute_p_keep(self, bar_beta_t, distance_map):
        d_norm = distance_map / self.d_global
        d_norm = torch.clamp(d_norm, 0.0, 1.0)
        sigma_norm = 0.1 
        spatial_decay = torch.exp(-(d_norm ** 2) / (2 * (sigma_norm ** 2)))
        clamped_decay = torch.clamp(spatial_decay, min=self.beta_min)
        
        return bar_beta_t * clamped_decay

class ExponentialDAUSTScheduler(BaseDAUSTScheduler):
    def _compute_p_keep(self, bar_beta_t, distance_map):
        # FIXED: Chuẩn hóa theo hằng số toàn cục để đảm bảo Translation Invariance
        d_norm = distance_map / self.d_global
        d_norm = torch.clamp(d_norm, 0.0, 1.0)
        
        g_d = 1.0 + self.gamma * (d_norm ** 2)
        
        p_keep = torch.pow(bar_beta_t, g_d)
        return torch.clamp(p_keep, min=self.beta_min)