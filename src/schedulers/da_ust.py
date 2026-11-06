import torch
import math

class BaseDAUSTScheduler:
    def __init__(self, num_train_timesteps=1000, sigma=5.0, beta_min=0.05,
                 gamma=5.0, image_size=256):
        self.num_train_timesteps = num_train_timesteps
        self.sigma = sigma
        self.beta_min = beta_min
        self.gamma = gamma
        self.image_size = image_size
        self.d_global = math.sqrt(image_size**2 + image_size**2)

        steps = torch.arange(num_train_timesteps + 1, dtype=torch.float32) / num_train_timesteps
        s = 0.008
        alphas = torch.cos(((steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        self.ust_alphas_cumprod = (alphas / alphas[0])[1:]

        self.gaussian_betas = torch.linspace(
            0.0001, 0.02, num_train_timesteps, dtype=torch.float32
        )
        self.gaussian_alphas_cumprod = torch.cumprod(
            1.0 - self.gaussian_betas, dim=0
        )

    def _compute_p_keep(self, bar_beta_t, distance_map, scale=None):
        raise NotImplementedError

    @staticmethod
    def _ensure_4d(x, name="tensor"):
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be a torch.Tensor")
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() == 4:
            return x
        raise ValueError(
            f"{name} must have shape (B,H,W) or (B,1,H,W), got {tuple(x.shape)}"
        )

    @staticmethod
    def _compute_visible_scale(visible_mask, min_scale=8.0):
        mv = BaseDAUSTScheduler._ensure_4d(visible_mask, "visible_mask").float()
        if mv.shape[1] != 1:
            raise ValueError("visible_mask must have one channel.")
        area = mv.sum(dim=(-2, -1))[:, 0]
        return torch.sqrt(area).clamp(min=float(min_scale))

    def add_noise(
        self,
        amodal_mask,
        distance_map,
        timesteps,
        visible_mask=None,
        scale=None,
        return_intermediate=False,
    ):
        device = amodal_mask.device
        amodal_mask = self._ensure_4d(amodal_mask, "amodal_mask")
        distance_map = self._ensure_4d(distance_map, "distance_map")

        if amodal_mask.shape != distance_map.shape:
            raise ValueError(
                f"amodal_mask and distance_map must match: "
                f"{tuple(amodal_mask.shape)} vs {tuple(distance_map.shape)}"
            )

        timesteps = torch.as_tensor(timesteps, device=device, dtype=torch.long)
        if timesteps.ndim == 0:
            timesteps = timesteps.reshape(1)
        if timesteps.ndim != 1 or timesteps.shape[0] != amodal_mask.shape[0]:
            raise ValueError("timesteps must have shape (B,)")

        ust_schedule = self.ust_alphas_cumprod.to(device)
        gaussian_schedule = self.gaussian_alphas_cumprod.to(device)

        bar_beta_t = ust_schedule[timesteps].view(-1, 1, 1, 1)
        gauss_alpha_bar_t = gaussian_schedule[timesteps].view(-1, 1, 1, 1)

        if scale is None and visible_mask is not None:
            scale = self._compute_visible_scale(visible_mask, min_scale=8.0)

        p_keep = self._compute_p_keep(bar_beta_t, distance_map, scale)

        if p_keep.shape != amodal_mask.shape:
            raise ValueError(
                f"p_keep and amodal_mask must match: "
                f"{tuple(p_keep.shape)} vs {tuple(amodal_mask.shape)}"
            )

        rand_tensor = torch.rand_like(amodal_mask)
        eroded_mask = (rand_tensor < p_keep).to(amodal_mask.dtype)

        modal_mask = (distance_map == 0).to(amodal_mask.dtype)
        x_prime_t = torch.where(
            modal_mask == 1,
            amodal_mask,
            eroded_mask * amodal_mask
        )

        epsilon = torch.randn_like(x_prime_t)
        x_t = (
            torch.sqrt(gauss_alpha_bar_t) * x_prime_t
            + torch.sqrt(1.0 - gauss_alpha_bar_t) * epsilon
        )

        if return_intermediate:
            return x_t, x_prime_t
        return x_t


class BaselineUSTScheduler(BaseDAUSTScheduler):
    def _compute_p_keep(self, bar_beta_t, distance_map, scale=None):
        return bar_beta_t.expand_as(distance_map)


class ClampedDAUSTScheduler(BaseDAUSTScheduler):
    def _compute_p_keep(self, bar_beta_t, distance_map, scale=None):
        d = distance_map.float()
        d_norm = (d / self.d_global).clamp(0.0, 1.0)
        sigma_norm = 0.1
        spatial_decay = torch.exp(
            -(d_norm ** 2) / (2.0 * sigma_norm ** 2)
        )
        clamped_decay = torch.clamp(spatial_decay, min=self.beta_min)
        return (bar_beta_t * clamped_decay).clamp(0.0, 1.0)


class ExponentialDAUSTScheduler(BaseDAUSTScheduler):
    """
    p_keep(p,t) = bar_beta_t ** (1 + gamma * d_norm(p)^2)

    d_norm(p) = clamp(distance_map(p) / scale, 0, 1)

    scale is computed only from M_v (default: sqrt(area(M_v))).
    No positive beta_min is applied to p_keep.
    """

    def __init__(
        self,
        num_train_timesteps=1000,
        sigma=5.0,
        beta_min=0.05,
        gamma=5.0,
        image_size=256,
        min_scale=8.0,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            sigma=sigma,
            beta_min=beta_min,
            gamma=gamma,
            image_size=image_size,
        )
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if min_scale <= 0:
            raise ValueError("min_scale must be > 0")
        self.gamma = float(gamma)
        self.min_scale = float(min_scale)

    def _compute_p_keep(self, bar_beta_t, distance_map, scale=None):
        if not torch.is_tensor(distance_map):
            raise TypeError("distance_map must be a torch.Tensor")

        d = distance_map.float()
        if d.dim() == 3:
            d = d.unsqueeze(1)
        elif d.dim() != 4:
            raise ValueError(
                "distance_map must have shape (B,H,W) or (B,1,H,W)"
            )

        beta = torch.as_tensor(
            bar_beta_t, dtype=torch.float32, device=d.device
        )
        if beta.ndim == 0:
            beta = beta.reshape(1, 1, 1, 1)
        elif beta.ndim == 1:
            beta = beta.reshape(-1, 1, 1, 1)
        elif beta.ndim != 4:
            raise ValueError(
                "bar_beta_t must be scalar, (B,), or (B,1,1,1)"
            )
        beta = beta.clamp(0.0, 1.0)

        if scale is None:
            raise ValueError(
                "ExponentialDAUSTScheduler requires scale computed from M_v. "
                "Pass visible_mask=... to add_noise(), or scale=... explicitly."
            )

        scale = torch.as_tensor(
            scale, dtype=torch.float32, device=d.device
        )
        if scale.ndim == 0:
            scale = scale.reshape(1, 1, 1, 1)
        elif scale.ndim == 1:
            scale = scale.reshape(-1, 1, 1, 1)
        elif scale.ndim != 4:
            raise ValueError(
                "scale must be scalar, (B,), or (B,1,1,1)"
            )

        B = d.shape[0]
        if beta.shape[0] not in (1, B):
            raise ValueError("beta batch dimension does not match distance_map")
        if scale.shape[0] not in (1, B):
            raise ValueError("scale batch dimension does not match distance_map")

        safe_scale = scale.clamp(min=self.min_scale)
        d_norm = (d / safe_scale).clamp(0.0, 1.0)
        g = 1.0 + self.gamma * d_norm.square()

        return beta.pow(g).clamp(0.0, 1.0)
