"""
denoising_net_v2.py — U-Net based Denoising Network for CondDiff-AMO.

LÝ DO TỒN TẠI FILE NÀY
======================

Diagnostic (dn_proposal_sanity_v3) cho thấy DenoisingNetwork cũ có
structural bottleneck: sensitivity của output với x_t chỉ ~10% khi PCN
features được giữ cố định. Nghĩa là nhánh x_t → e1 → e2 → e3 → bottleneck
trong DN cũ KHÔNG propagate x_t tới output đủ mạnh.

Nguyên nhân kiến trúc: trong DenoisingNetwork cũ, x_t chỉ đi vào 1 đường
duy nhất qua bottleneck, phải cạnh tranh với PCN features (A1) trong 1
phép conv duy nhất. Không có skip connection để rescue x_t signal.

UNetDenoisingNetwork giải quyết bằng cách:
  1. Nhận cat([x_t, M_v]) làm input trực tiếp (2 channels).
  2. U-Net encoder-decoder với skip connections: d1_up + e2, d2_up + e1.
  3. Time embedding inject ở MỌI tầng encoder + decoder.

LƯU Ý QUAN TRỌNG
================

Test v3 (random init) cho thấy NEW DN có image sensitivity rất thấp
(0.0023 vs 0.351 của OLD DN). Đây có thể là artifact của random init
(cần verify sau khi train), hoặc có thể là architectural issue thật sự.
KHÔNG nên train full 40 epochs NEW DN trước khi chạy controlled
mini-run 3 epochs để verify learning signal.

Xem file dn_proposal_sanity_v3_analysis.md để có đầy đủ context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import các helper module có sẵn từ denoising_net.py gốc
from src.models.pcn_extractor import TimeEmbedding
from src.models.denoising_net import (
    LocalEmphasis,
    AdaptiveFeatureGate,
    OcclusionAwareContextRefinement,
)


class UNetDenoisingNetwork(nn.Module):
    """
    U-Net based Denoising Network cho CondDiff-AMO.

    Khác biệt chính so với DenoisingNetwork cũ:
      - Input: cat([x_t, M_v]) = 2 channels (cũ: chỉ x_t = 1 channel)
      - Decoder có skip connections: d1_up + e2, d2_up + e1
      - Time injection ở mọi tầng (encoder + decoder)
      - Thêm ~370K params so với DN cũ (+3.5%)

    Cùng sử dụng PCN features qua đường A1 (OCR → LE → conv_A3 → conv_A2 → AFG)
    giống hệt DN cũ, chỉ khác nhánh U-Net của x_t.

    Args:
        pcn_channels (list[int]): số kênh của [F1, F2, F3, F4] từ PCN.
        fuse_channels (int): số kênh chuẩn hoá cho PCN fusion path.
        use_ocr (bool): bật/tắt OCR module (ablation).
    """

    def __init__(
        self,
        pcn_channels=[64, 128, 320, 512],
        fuse_channels=256,
        use_ocr=True,
    ):
        super().__init__()
        self.use_ocr = use_ocr

        # ============================================
        # PCN fusion path (giống DN cũ)
        # ============================================
        if self.use_ocr:
            self.ocr_module = OcclusionAwareContextRefinement(pcn_channels[-1])

        self.le_module = LocalEmphasis(pcn_channels, fuse_channels)

        self.conv_A3 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_A2 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.afg_module = AdaptiveFeatureGate(fuse_channels)

        # ============================================
        # U-Net encoder (2 input channels: x_t, M_v)
        # ============================================
        # Encoder spatial: 256 → 128 → 64
        self.enc1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)          # 256x256
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)   # 128x128
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 64x64

        # Bottleneck nhận PCN features (A1) tại spatial 64x64
        self.bottleneck = nn.Conv2d(256 + fuse_channels, 256, kernel_size=3, padding=1)

        # ============================================
        # U-Net decoder với skip connections
        # ============================================
        # Quan trọng: upsample TRƯỚC, concat skip SAU.
        # Nếu concat trước upsample, spatial dims sẽ mismatch (bug v1 của Gemini).
        self.dec1_up = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 64 → 128
        self.dec1_fuse = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)              # + e2

        self.dec2_up = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 128 → 256
        self.dec2_fuse = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)                 # + e1

        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # ============================================
        # Time embedding + injection
        # ============================================
        self.time_embed = TimeEmbedding(embed_dim=256)

        unet_channels = {
            "enc1": 64,
            "enc2": 128,
            "enc3": 256,
            "bottleneck": 256,
            "dec1": 128,
            "dec2": 64,
        }
        self.time_projs = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(256, c), nn.SiLU())
            for name, c in unet_channels.items()
        })

    def forward(self, x_t, t, pcn_features, M_v):
        """
        Args:
            x_t: (B, 1, H, W) — noisy mask tại timestep t.
            t: (B,) — timestep tensor.
            pcn_features: list [F1, F2, F3, F4] từ PCN.
            M_v: (B, 1, H, W) — visible mask (điều kiện bổ sung).

        Returns:
            x_hat_0: (B, 1, H, W) — logits của dự đoán mask sạch (chưa sigmoid).
        """
        F1, F2, F3, F4 = pcn_features

        # ============================================
        # 1. PCN fusion path → A1 (giống DN cũ)
        # ============================================
        if self.use_ocr:
            F4 = self.ocr_module(F4)

        F_up1, F_up2, F_up3, F_up4 = self.le_module([F1, F2, F3, F4])

        A4 = F_up4
        A3 = self.conv_A3(torch.cat([A4, F_up3], dim=1))
        A2 = self.conv_A2(torch.cat([A3, F_up2], dim=1))
        A1 = self.afg_module(A2, F_up1)  # (B, fuse_channels, 64, 64)

        # ============================================
        # 2. Time embedding
        # ============================================
        t_emb = self.time_embed(t)  # (B, 256)

        def inject_t(feat, name):
            scale = self.time_projs[name](t_emb).unsqueeze(-1).unsqueeze(-1)
            return feat + scale

        # ============================================
        # 3. U-Net encoder
        # ============================================
        # Direct input: cat([x_t, M_v])
        unet_input = torch.cat([x_t, M_v], dim=1)  # (B, 2, 256, 256)

        e1 = F.relu(inject_t(self.enc1(unet_input), "enc1"))    # (B, 64, 256, 256)
        e2 = F.relu(inject_t(self.enc2(e1), "enc2"))            # (B, 128, 128, 128)
        e3 = F.relu(inject_t(self.enc3(e2), "enc3"))            # (B, 256, 64, 64)

        # ============================================
        # 4. Bottleneck: concat PCN features
        # ============================================
        b = F.relu(inject_t(
            self.bottleneck(torch.cat([e3, A1], dim=1)),
            "bottleneck",
        ))  # (B, 256, 64, 64)

        # ============================================
        # 5. Decoder với skip connections
        # ============================================
        # Skip 1: upsample b → concat e2
        d1_up = F.relu(inject_t(self.dec1_up(b), "dec1"))       # (B, 128, 128, 128)
        d1 = F.relu(self.dec1_fuse(torch.cat([d1_up, e2], dim=1)))  # (B, 128, 128, 128)

        # Skip 2: upsample d1 → concat e1
        d2_up = F.relu(inject_t(self.dec2_up(d1), "dec2"))      # (B, 64, 256, 256)
        d2 = F.relu(self.dec2_fuse(torch.cat([d2_up, e1], dim=1)))  # (B, 64, 256, 256)

        x_hat_0 = self.final_conv(d2)  # (B, 1, 256, 256)

        return x_hat_0


# ============================================================
# FACTORY — chuyển đổi giữa OLD và NEW DN
# ============================================================
def build_denoising_network(
    variant: str,
    pcn_channels,
    fuse_channels=256,
    use_ocr=True,
):
    """
    Factory để build DN theo biến thể.

    Args:
        variant: "old" | "new"
        pcn_channels: channel list từ PCN.
        fuse_channels: số kênh fusion.
        use_ocr: bật/tắt OCR.

    Returns:
        nn.Module với forward tương ứng.

    LƯU Ý: interface forward KHÁC NHAU giữa 2 variant:
        - old: dn(x_t, t, pcn_features)
        - new: dn(x_t, t, pcn_features, M_v)
    Caller phải biết variant để truyền đúng args.
    """
    if variant == "old":
        from src.models.denoising_net import DenoisingNetwork
        return DenoisingNetwork(
            pcn_channels=pcn_channels,
            fuse_channels=fuse_channels,
            use_ocr=use_ocr,
        )
    elif variant == "new":
        return UNetDenoisingNetwork(
            pcn_channels=pcn_channels,
            fuse_channels=fuse_channels,
            use_ocr=use_ocr,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'old' or 'new'.")


# ============================================================
# UNIT TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Unit test: UNetDenoisingNetwork")
    print("=" * 60)

    B = 2
    pcn_channels = [64, 128, 320, 512]
    F1 = torch.rand(B, pcn_channels[0], 64, 64)
    F2 = torch.rand(B, pcn_channels[1], 32, 32)
    F3 = torch.rand(B, pcn_channels[2], 16, 16)
    F4 = torch.rand(B, pcn_channels[3], 8, 8)
    x_t = torch.rand(B, 1, 256, 256)
    M_v = torch.rand(B, 1, 256, 256)
    t = torch.tensor([200, 800])

    dn_new = UNetDenoisingNetwork(pcn_channels=pcn_channels, fuse_channels=256, use_ocr=True)
    out = dn_new(x_t, t, [F1, F2, F3, F4], M_v)

    print(f"Input x_t: {tuple(x_t.shape)}")
    print(f"Input M_v: {tuple(M_v.shape)}")
    print(f"Output   : {tuple(out.shape)}")
    print(f"Finite   : {torch.isfinite(out).all().item()}")
    print(f"Params   : {sum(p.numel() for p in dn_new.parameters()):,}")

    assert out.shape == (B, 1, 256, 256), "Output shape mismatch!"

    # ---- Shape check cho decoder skip connections ----
    print("\n--- Checking decoder skip connections ---")
    with torch.no_grad():
        e1 = dn_new.enc1(torch.cat([x_t, M_v], dim=1))
        e2 = dn_new.enc2(F.relu(e1))
        e3 = dn_new.enc3(F.relu(e2))
        print(f"  e1: {tuple(e1.shape)}  (256x256, 64ch)")
        print(f"  e2: {tuple(e2.shape)}  (128x128, 128ch)")
        print(f"  e3: {tuple(e3.shape)}  (64x64, 256ch)")

    print("\n✅ All tests passed.")