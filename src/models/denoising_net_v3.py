import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pcn_extractor import TimeEmbedding

class TimeFiLMResBlock(nn.Module):
    """
    Residual block được điều kiện hóa bởi timestep thông qua FiLM.
    Input/output: [B, C, H, W]
    """
    def __init__(self, in_channels, out_channels, time_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_affine = nn.Linear(time_dim, out_channels * 2)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

        # Ban đầu FiLM không làm biến đổi feature quá mạnh.
        nn.init.zeros_(self.time_affine.weight)
        nn.init.zeros_(self.time_affine.bias)

    def forward(self, x, t_emb):
        residual = self.skip(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        gamma, beta = self.time_affine(t_emb).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        h = h * (1.0 + gamma) + beta
        return F.silu(h + residual)


class DNv3(nn.Module):
    """
    Multi-Scale Conditioned Residual U-Net.

    Inputs:
        x_t:         [B, 1, H, W]
        t:           [B]
        pcn_features: tuple/list gồm 4 feature maps
        M_v:         [B, 1, H, W]

    Output:
        Amodal-mask logits [B, 1, H, W]
    """
    def __init__(
        self,
        pcn_channels=(64, 128, 320, 512),
        fuse_channels=256,
        use_ocr=True,
    ):
        super().__init__()

        self.use_ocr = use_ocr

        if use_ocr:
            self.ocr_module = OcclusionAwareContextRefinement(
                pcn_channels[-1]
            )

        self.le_module = LocalEmphasis(
            list(pcn_channels), fuse_channels
        )
        self.conv_A3 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_A2 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.afg_module = AdaptiveFeatureGate(fuse_channels)

        # Timestep embedding
        self.time_embed = TimeEmbedding(embed_dim=256)

        # Separate path for x_t and visible mask.
        self.input_proj = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.enc1 = TimeFiLMResBlock(64, 64)

        self.down2 = nn.Conv2d(64, 128, kernel_size=3,
                               stride=2, padding=1)
        self.enc2 = TimeFiLMResBlock(128, 128)

        self.down3 = nn.Conv2d(128, 256, kernel_size=3,
                               stride=2, padding=1)
        self.enc3 = TimeFiLMResBlock(256, 256)

        # PCN condition is injected at multiple decoder/encoder levels.
        self.cond_e3 = nn.Conv2d(fuse_channels, 256, kernel_size=1)
        self.cond_mid = nn.Conv2d(fuse_channels, 256, kernel_size=1)
        self.cond_d1 = nn.Conv2d(fuse_channels, 128, kernel_size=1)
        self.cond_d2 = nn.Conv2d(fuse_channels, 64, kernel_size=1)

        self.bottleneck = TimeFiLMResBlock(256, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )
        self.fuse_d1 = nn.Conv2d(128 + 128, 128, kernel_size=1)
        self.dec1 = TimeFiLMResBlock(128, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.fuse_d2 = nn.Conv2d(64 + 64, 64, kernel_size=1)
        self.dec2 = TimeFiLMResBlock(64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    @staticmethod
    def resize_condition(condition, target):
        return F.interpolate(
            condition,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x_t, t, pcn_features, M_v):
        F1, F2, F3, F4 = pcn_features

        if self.use_ocr:
            F4 = self.ocr_module(F4)

        F_up1, F_up2, F_up3, F_up4 = self.le_module(
            [F1, F2, F3, F4]
        )

        A3 = self.conv_A3(torch.cat([F_up4, F_up3], dim=1))
        A2 = self.conv_A2(torch.cat([A3, F_up2], dim=1))
        A1 = self.afg_module(A2, F_up1)

        t_emb = self.time_embed(t)

        # Encoder: retain spatial details from x_t and M_v.
        x = torch.cat([x_t, M_v], dim=1)
        e1 = self.enc1(self.input_proj(x), t_emb)

        e2 = self.enc2(self.down2(e1), t_emb)

        e3 = self.down3(e2)
        c3 = self.resize_condition(A1, e3)
        e3 = self.enc3(e3 + self.cond_e3(c3), t_emb)

        # Bottleneck receives image-conditioned features again.
        c_mid = self.resize_condition(A1, e3)
        mid = self.bottleneck(
            e3 + self.cond_mid(c_mid), t_emb
        )

        # Decoder: skip connections preserve localization.
        d1_up = self.up2(mid)
        if d1_up.shape[-2:] != e2.shape[-2:]:
            d1_up = F.interpolate(
                d1_up, size=e2.shape[-2:],
                mode="bilinear", align_corners=False
            )

        d1 = self.fuse_d1(torch.cat([d1_up, e2], dim=1))
        c_d1 = self.resize_condition(A1, d1)
        d1 = self.dec1(
            d1 + self.cond_d1(c_d1), t_emb
        )

        d2_up = self.up1(d1)
        if d2_up.shape[-2:] != e1.shape[-2:]:
            d2_up = F.interpolate(
                d2_up, size=e1.shape[-2:],
                mode="bilinear", align_corners=False
            )

        d2 = self.fuse_d2(torch.cat([d2_up, e1], dim=1))
        c_d2 = self.resize_condition(A1, d2)
        d2 = self.dec2(
            d2 + self.cond_d2(c_d2), t_emb
        )

        logits = self.final_conv(d2)

        # Bảo đảm đầu ra khớp kích thước đầu vào.
        if logits.shape[-2:] != x_t.shape[-2:]:
            logits = F.interpolate(
                logits, size=x_t.shape[-2:],
                mode="bilinear", align_corners=False
            )

        return logits
