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

# class ExponentialDAUSTScheduler(BaseDAUSTScheduler):
#     def _compute_p_keep(self, bar_beta_t, distance_map):
#         # FIXED: Chuẩn hóa theo hằng số toàn cục để đảm bảo Translation Invariance
#         d_norm = distance_map / self.d_global
#         d_norm = torch.clamp(d_norm, 0.0, 1.0)
        
#         g_d = 1.0 + self.gamma * (d_norm ** 2)
        
#         p_keep = torch.pow(bar_beta_t, g_d)
#         return torch.clamp(p_keep, min=self.beta_min)

class ExponentialDAUSTScheduler(BaseDAUSTScheduler):
    """
    Spatially adaptive UST scheduler.

    p_keep(p, t) =
        beta_bar_t ** (1 + gamma * d_norm(p)^2)

    where

        d_norm(p) = clamp(distance_map(p) / scale, 0, 1)

    `scale` must be computed from Mv only, so that the same
    rule is available during training and inference.
    """

    def __init__(
        self,
        gamma: float = 5.0,
        min_scale: float = 8.0,
    ):
        super().__init__()

        if gamma < 0:
            raise ValueError("gamma must be >= 0")

        if min_scale <= 0:
            raise ValueError("min_scale must be > 0")

        self.gamma = float(gamma)
        self.min_scale = float(min_scale)

    def _compute_p_keep(
        self,
        bar_beta_t,
        distance_map: torch.Tensor,
        scale,
    ) -> torch.Tensor:
        """
        Args:
            bar_beta_t:
                Scalar or tensor of shape (B,).

            distance_map:
                Shape (B,H,W) or (B,1,H,W).
                Distance to the visible region.

            scale:
                Scalar or tensor of shape (B,).
                Object-specific distance scale computed ONLY from Mv.

        Returns:
            p_keep:
                Shape (B,1,H,W), or matching the normalized
                distance-map representation.
        """

        if not torch.is_tensor(distance_map):
            raise TypeError("distance_map must be a torch.Tensor")

        # Always perform this computation in float32.
        d = distance_map.float()

        # Normalize shape to (B,1,H,W).
        if d.dim() == 3:
            d = d.unsqueeze(1)

        elif d.dim() != 4:
            raise ValueError(
                "distance_map must have shape "
                "(B,H,W) or (B,1,H,W)"
            )

        beta = torch.as_tensor(
            bar_beta_t,
            dtype=torch.float32,
            device=d.device,
        )

        scale = torch.as_tensor(
            scale,
            dtype=torch.float32,
            device=d.device,
        )

        # ---------------------------------------------------------
        # Normalize beta -> (B,1,1,1)
        # ---------------------------------------------------------

        if beta.ndim == 0:
            beta = beta.reshape(1, 1, 1, 1)

        elif beta.ndim == 1:
            beta = beta.reshape(-1, 1, 1, 1)

        else:
            raise ValueError(
                "bar_beta_t must be a scalar or have shape (B,)"
            )

        # ---------------------------------------------------------
        # Normalize scale -> (B,1,1,1)
        # ---------------------------------------------------------

        if scale.ndim == 0:
            scale = scale.reshape(1, 1, 1, 1)

        elif scale.ndim == 1:
            scale = scale.reshape(-1, 1, 1, 1)

        else:
            raise ValueError(
                "scale must be a scalar or have shape (B,)"
            )

        # ---------------------------------------------------------
        # Clamp beta.
        #
        # No beta_min is introduced here.
        # This preserves the terminal behavior of the schedule.
        # ---------------------------------------------------------

        beta = beta.clamp(0.0, 1.0)

        # Prevent division by zero without Python-side branching.
        safe_scale = scale.clamp(min=self.min_scale)

        # ---------------------------------------------------------
        # Spatial normalization
        # ---------------------------------------------------------

        d_norm = d / safe_scale
        d_norm = d_norm.clamp(0.0, 1.0)

        # ---------------------------------------------------------
        # Distance-dependent exponent
        # ---------------------------------------------------------

        g = 1.0 + self.gamma * d_norm.square()

        # ---------------------------------------------------------
        # DA-UST
        # ---------------------------------------------------------

        p_keep = beta.pow(g)

        # Numerical safety.
        return p_keep.clamp(0.0, 1.0)