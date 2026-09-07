import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pcn_extractor import TimeEmbedding

# ==========================================
# 0. MODULE: CHANNEL ATTENTION (dùng nội bộ trong OCR)
# ==========================================
class ChannelAttention(nn.Module):
    """
    SE-style channel attention. Dùng để reweight kênh nào quan trọng
    (occlusion-relevant) trong mỗi nhánh của OCR.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))  # (B, C, 1, 1)
        return x * w


# ==========================================
# 0b. MODULE: PYRAMID ATTENTION (multi-scale sampling)
# ==========================================
class PyramidAttention(nn.Module):
    """
    Pyramid Attention đúng nghĩa "multi-scale sampling": mô phỏng occlusion ở
    NHIỀU kích cỡ khác nhau bằng cách average-pool cùng một feature map với
    nhiều kernel size (mặc định 3, 5, 7), rồi mới trích xuất + gộp lại thành
    một attention map duy nhất.

    Khác với Spatial Attention kiểu CBAM (chỉ 1 conv 7x7 duy nhất trên feature
    gốc): ở đây mỗi scale được XỬ LÝ RIÊNG (pool -> conv giảm kênh) trước khi
    gộp, nên attention map thực sự tổng hợp thông tin từ nhiều "cửa sổ nhìn"
    khác nhau (giống occlusion có thể to hoặc nhỏ tuỳ vật thể).

    Input : x  (B, C, H, W)
    Output: attn (B, 1, H, W) — trong khoảng (0, 1), dùng để reweight x
    """
    def __init__(self, channels, pool_sizes=(3, 5, 7), reduced_channels=None):
        super().__init__()
        self.pool_sizes = pool_sizes
        reduced_channels = reduced_channels or max(channels // 4, 16)

        # Mỗi scale: pool (giữ nguyên H,W nhờ padding=k//2, stride=1) -> conv 1x1 giảm kênh
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, kernel_size=1),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True)
            ) for _ in pool_sizes
        ])

        # Gộp các scale lại thành 1 attention map
        self.fuse = nn.Sequential(
            nn.Conv2d(reduced_channels * len(pool_sizes), 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_feats = []
        for k, conv in zip(self.pool_sizes, self.scale_convs):
            # avg_pool với stride=1, padding=k//2 -> giữ nguyên kích thước không gian
            pooled = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
            multi_scale_feats.append(conv(pooled))

        fused = torch.cat(multi_scale_feats, dim=1)   # (B, reduced_channels * len(pool_sizes), H, W)
        attn = self.fuse(fused)                        # (B, 1, H, W)
        return attn


# ==========================================
# 1. MODULE: OCCLUSION-AWARE CONTEXT REFINEMENT (OCR)
# ==========================================
class OcclusionAwareContextRefinement(nn.Module):
    """
    Tinh chỉnh riêng tầng đặc trưng sâu nhất F4 (Fig. 3B, phần OCR trong paper).

    Gồm 3 nhánh chạy song song trên F4:
      - Local Branch  : dilation nhỏ (mặc định 1, 2)  -> bắt chi tiết biên (occlusion boundaries)
      - Global Branch : dilation lớn hơn (mặc định 3, 4) -> bắt ngữ cảnh tầm xa (long-range)
      - Pooling Branch: global average pooling -> ngữ cảnh toàn cục (global scene semantics)
    Mỗi nhánh có Channel Attention riêng để nhấn mạnh kênh liên quan tới occlusion.
    Sau khi fuse 3 nhánh, thêm Pyramid Attention (multi-scale pooling 3x3/5x5/7x7,
    xem class PyramidAttention) để dynamically reweight vùng không gian nào quan
    trọng cho việc suy luận vùng bị che khuất, mô phỏng occlusion ở nhiều kích cỡ.

    LƯU Ý QUAN TRỌNG VỀ DILATION: F4 thường chỉ (H,W)=(8,8) với ảnh 256x256 (stride 32).
    Dilation rate PHẢI tỷ lệ với kích thước feature map thực tế -- dilation quá lớn so với
    map (vd 6, 12 copy nguyên từ ASPP thiết kế cho map 32-64x64) khiến 2 tap biên của kernel
    3x3 rơi hoàn toàn ra ngoài map (vào vùng padding=0) ở MỌI vị trí, khiến conv thoái hoá
    gần như 1x1 (không còn "long-range dependency" như tên gọi). Mặc định mới (3,4) đảm bảo
    tap biên (cách tâm 2*dilation = 6, 8 pixel) vẫn còn nằm trong map 8x8 ở phần lớn vị trí.

    Input : F4  (B, C, H, W)   — feature sâu nhất từ PVT (thường H=W=8 với ảnh 256x256)
    Output: F4' (B, C, H, W)   — cùng shape, đã được tinh chỉnh (residual connection)
    """
    def __init__(self, channels, local_dilations=(1, 2), global_dilations=(3, 4)):
        super().__init__()
        branch_ch = max(channels // 2, 32)

        def conv_layer(in_ch, out_ch, dilation):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        d_local_1, d_local_2 = local_dilations
        d_global_1, d_global_2 = global_dilations

        # Local Branch: fine-grained textures / occlusion boundaries
        self.local_branch = nn.Sequential(
            conv_layer(channels, branch_ch, dilation=d_local_1),
            conv_layer(branch_ch, branch_ch, dilation=d_local_2),
            ChannelAttention(branch_ch)
        )

        # Global Branch: long-range dependencies (dilation vừa đủ để KHÔNG rơi hết vào padding
        # trên feature map nhỏ -- xem giải thích ở docstring phía trên)
        self.global_branch = nn.Sequential(
            conv_layer(channels, branch_ch, dilation=d_global_1),
            conv_layer(branch_ch, branch_ch, dilation=d_global_2),
            ChannelAttention(branch_ch)
        )

        # Pooling Branch: high-level global context
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_ch, kernel_size=1),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            ChannelAttention(branch_ch)
        )

        # Fuse 3 nhánh lại về đúng số kênh gốc
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Pyramid Attention: multi-scale pooling (3x3, 5x5, 7x7) rồi mới gộp thành
        # attention map -> mô phỏng occlusion ở nhiều kích thước khác nhau.
        self.pyramid_attn = PyramidAttention(channels, pool_sizes=(3, 5, 7))

    def forward(self, F4: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F4.shape

        local_out = self.local_branch(F4)                        # (B, branch_ch, H, W)
        global_out = self.global_branch(F4)                      # (B, branch_ch, H, W)

        pool_out = self.pool_branch(F4)                           # (B, branch_ch, 1, 1)
        pool_out = F.interpolate(pool_out, size=(H, W), mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([local_out, global_out, pool_out], dim=1))  # (B, C, H, W)

        attn = self.pyramid_attn(fused)                            # (B, 1, H, W)

        # Residual: giữ lại thông tin gốc của F4, cộng thêm phần đã được refine + attention
        out = F4 + fused * attn                                    # (B, C, H, W)
        return out


# ==========================================
# 2. MODULE: LOCAL EMPHASIS (LE)
# ==========================================
class LocalEmphasis(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        """
        Đồng bộ hóa các khối đặc trưng đa cấp độ (F1, F2, F3, F4) về cùng một
        kích thước không gian (bằng với F1) và cùng số lượng kênh (out_channels).
        """
        super().__init__()
        # Tạo danh sách các lớp 1x1 Conv để nén/phóng số kênh về chuẩn 256
        self.projs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: List gồm [F1, F2, F3, F4] với kích thước giảm dần.
        Returns:
            List [F_up1, F_up2, F_up3, F_up4] có cùng shape: (B, 256, H/4, W/4)
        """
        F1_size = features[0].shape[2:]  # Lấy (Height, Width) của F1 làm mốc
        f_up = []

        for i, f in enumerate(features):
            p = self.projs[i](f)  # Đưa về cùng số channels
            if i > 0:
                # Phóng to các đặc trưng sâu (F2, F3, F4) lên bằng F1
                p = F.interpolate(p, size=F1_size, mode='bilinear', align_corners=False)
            f_up.append(p)

        return f_up


# ==========================================
# 3. MODULE: ADAPTIVE FEATURE GATE (AFG)
# ==========================================
class AdaptiveFeatureGate(nn.Module):
    def __init__(self, channels=256):
        """Cơ chế cổng kiểm soát (Gate) để dung hợp đặc trưng sâu (A2) và nông (F_up1)."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)  # Xuất ra 1 kênh làm tỷ lệ (Gate)

    def forward(self, A2, F_up1):
        """
        A2: Đặc trưng ngữ nghĩa cấp cao (Shape: B, 256, H/4, W/4)
        F_up1: Đặc trưng chi tiết cấp thấp (Shape: B, 256, H/4, W/4)
        """
        concat_feat = torch.cat([A2, F_up1], dim=1)  # (B, 512, H/4, W/4)

        # Gate = Sigmoid(Conv(ReLU(Conv(concat(A2, F_up1)))))
        gate = torch.sigmoid(self.conv2(F.relu(self.conv1(concat_feat))))  # (B, 1, H/4, W/4)

        # A1 = Gate * A2 + (1 - Gate) * F_up1
        A1 = gate * A2 + (1.0 - gate) * F_up1  # (B, 256, H/4, W/4)
        return A1


# ==========================================
# 4. MODULE: DENOISING NETWORK (DN - MẠNG CHÍNH)
# ==========================================
class DenoisingNetwork(nn.Module):
    def __init__(self, pcn_channels=[64, 128, 320, 512], fuse_channels=256, use_ocr=True):
        """
        Args:
            pcn_channels: số kênh của [F1, F2, F3, F4] lấy từ PCNExtractor.feature_channels.
            fuse_channels: số kênh chuẩn hóa dùng trong LE/AFG/Bottleneck (mặc định 256).
            use_ocr: bật/tắt module OCR để tiện làm ablation (Table 4 trong paper).
        """
        super().__init__()
        self.use_ocr = use_ocr

        # OCR chỉ tác động lên F4 (tầng sâu nhất), trước khi đưa vào LE
        if self.use_ocr:
            self.ocr_module = OcclusionAwareContextRefinement(pcn_channels[-1])

        self.le_module = LocalEmphasis(pcn_channels, fuse_channels)

        # Lớp chập cho Progressive Fusion
        self.conv_A3 = nn.Sequential(nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1), nn.ReLU())
        self.conv_A2 = nn.Sequential(nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1), nn.ReLU())

        self.afg_module = AdaptiveFeatureGate(fuse_channels)

        # ==========================================
        # LIGHTWEIGHT U-NET ENCODER-DECODER
        # ==========================================
        # Đầu vào: Mask nhiễu x_t (1 kênh)
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)   # Xuống H/2
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # Xuống H/4 (Bằng với A1)

        # Nơi Điều kiện (A1) chèn vào mạng khử nhiễu
        self.bottleneck = nn.Conv2d(256 + fuse_channels, 256, 3, padding=1)

        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # Lên H/2
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)   # Lên H
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)  # Trả về 1 kênh mask (x_hat_0)

        # ==========================================
        # FIX: TIMESTEP CONDITIONING CHO NHÁNH U-NET
        # ==========================================
        # Trước đây `t_emb` được nhận vào forward() nhưng KHÔNG hề được sử dụng ở đâu cả --
        # nhánh U-Net xử lý trực tiếp x_t (enc1->enc2->enc3->bottleneck->dec1->dec2) hoàn
        # toàn không biết đang ở timestep nào, ngoại trừ gián tiếp qua A1 (vốn đã mang thông
        # tin t từ PCN). Đây là thiếu sót so với thiết kế diffusion U-Net chuẩn (thường tiêm
        # timestep ở MỌI tầng, không chỉ 1 điểm). Thêm time embedding + projection riêng cho
        # từng tầng, tái dùng đúng TimeEmbedding đã có trong pcn_extractor.py để nhất quán.
        self.time_embed = TimeEmbedding(embed_dim=256)
        unet_channels = {"enc1": 64, "enc2": 128, "enc3": 256, "bottleneck": 256, "dec1": 128, "dec2": 64}
        self.time_projs = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(256, c), nn.SiLU())
            for name, c in unet_channels.items()
        })

    def forward(self, x_t, t, pcn_features):
        """
        x_t: Mask nhiễu (B, 1, H, W)
        t_emb: Timestep embedding (không gian tùy chỉnh nếu cần)
        pcn_features: [F1, F2, F3, F4]
        """
        F1, F2, F3, F4 = pcn_features

        # 0. Occlusion-Aware Context Refinement (chỉ trên F4)
        if self.use_ocr:
            F4 = self.ocr_module(F4)

        # 1. Local Emphasis
        F_up1, F_up2, F_up3, F_up4 = self.le_module([F1, F2, F3, F4])

        # 2. Progressive Fusion (Dung hợp dần từ sâu ra nông)
        A4 = F_up4
        A3 = self.conv_A3(torch.cat([A4, F_up3], dim=1))
        A2 = self.conv_A2(torch.cat([A3, F_up2], dim=1))

        # 3. Adaptive Feature Gate
        A1 = self.afg_module(A2, F_up1)  # Shape: (B, 256, H/4, W/4)

        # 4. Time embedding dùng chung cho toàn bộ nhánh U-Net (FIX: trước đây không dùng)
        t_emb = self.time_embed(t)  # (B, 256)

        def inject_t(feat, name):
            scale = self.time_projs[name](t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            return feat + scale

        # 5. Lightweight U-Net, tiêm timestep ở MỌI tầng
        e1 = F.relu(inject_t(self.enc1(x_t), "enc1"))
        e2 = F.relu(inject_t(self.enc2(e1), "enc2"))
        e3 = F.relu(inject_t(self.enc3(e2), "enc3"))  # Shape: (B, 256, H/4, W/4)

        # Conditioning: Ghép A1 vào cổ chai (Bottleneck)
        bottleneck_input = torch.cat([e3, A1], dim=1)  # Shape: (B, 512, H/4, W/4)
        b = F.relu(inject_t(self.bottleneck(bottleneck_input), "bottleneck"))

        d1 = F.relu(inject_t(self.dec1(b), "dec1"))
        d2 = F.relu(inject_t(self.dec2(d1), "dec2"))
        x_hat_0 = self.final_conv(d2)  # Shape: (B, 1, H, W) (Chưa qua Sigmoid để tính BCE Loss bằng Logits)

        return x_hat_0


# ==========================================
# 5. MODULE: HÀM MẤT MÁT (LOSS FUNCTIONS)
# ==========================================
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, target_mask, weight_map=None):
        """pred_logits: Kết quả thô từ mạng chưa qua Sigmoid."""
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask, reduction='none')
        if weight_map is not None:
            loss = loss * weight_map
        return loss.mean()


class WeightedIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, target_mask, weight_map=None):
        pred_probs = torch.sigmoid(pred_logits)

        if weight_map is None:
            weight_map = torch.ones_like(target_mask)

        # Áp dụng trọng số vào cả intersection và union
        intersection = (pred_probs * target_mask * weight_map).sum(dim=(2, 3))
        union = ((pred_probs + target_mask) * weight_map).sum(dim=(2, 3)) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return (1.0 - iou).mean()


class BoundaryAwareEdgeLoss(nn.Module):
    """
    Edge Loss thiết kế riêng cho Amodal Segmentation.

    Điểm khác biệt so với Edge Loss thông thường:
    - Dùng Laplacian thay vì Sobel: nhạy hơn với cả 4 hướng
    - Spatial weighting: phạt nặng hơn ở vùng biên visible/occluded
      (nơi mạng hay mắc lỗi nhất)
    - Smooth L1 thay vì L2: robust hơn với annotation noise
    """
    def __init__(self, weight_boundary: float = 2.0):
        """
        Args:
            weight_boundary: Hệ số phóng đại loss tại vùng biên visible/occluded.
                             2.0 = phạt vùng biên gấp đôi vùng còn lại.
        """
        super().__init__()
        self.weight_boundary = weight_boundary

        # Laplacian kernel — detect edge theo cả 4 hướng
        laplacian = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

    def _extract_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Tính edge map từ binary mask bằng Laplacian.
        Input:  (B, 1, H, W) — float, giá trị [0, 1]
        Output: (B, 1, H, W) — edge magnitude, đã clamp về [0, 1]
        """
        edges = F.conv2d(mask, self.laplacian, padding=1)
        return edges.abs().clamp(0, 1)

    def forward(
        self,
        pred_logits: torch.Tensor,   # (B, 1, H, W) — raw logits từ DN
        target_mask: torch.Tensor,   # (B, 1, H, W) — ground truth M_a, binary
        modal_mask: torch.Tensor,    # (B, 1, H, W) — M_v, để tính spatial weight
    ) -> torch.Tensor:

        # 1. Soft prediction (differentiable)
        pred_probs = torch.sigmoid(pred_logits)

        # 2. Edge map của prediction và GT
        pred_edges = self._extract_edges(pred_probs)
        target_edges = self._extract_edges(target_mask)

        # 3. Spatial weight map:
        #    - Vùng biên giữa visible và occluded (d(M_v) nhỏ): weight cao
        #    - Vùng sâu trong occluded hoặc background: weight thấp hơn
        #
        #    Proxy: dùng chính target_edges của M_v làm boundary indicator
        #    Pixel gần biên M_v → đây là nơi transition visible/occluded
        modal_edges = self._extract_edges(modal_mask)  # Edge của visible mask
        # Dilate nhẹ để tạo vùng ảnh hưởng quanh biên
        weight_map = 1.0 + (self.weight_boundary - 1.0) * modal_edges

        # 4. Smooth L1 loss trên edge maps, có spatial weight
        edge_diff = F.smooth_l1_loss(pred_edges, target_edges, reduction='none')
        weighted_loss = (edge_diff * weight_map).mean()

        return weighted_loss


# ==========================================
# 6. KHỐI KIỂM THỬ CỤC BỘ
# ==========================================
if __name__ == "__main__":
    print("🚀 Khởi động Unit Test cho DenoisingNetwork (có OCR)...")

    # Giả lập 4 tầng đặc trưng như output của PVTv2-B4 với ảnh 256x256
    pcn_channels = [64, 128, 320, 512]
    B = 2
    F1 = torch.rand(B, pcn_channels[0], 64, 64)
    F2 = torch.rand(B, pcn_channels[1], 32, 32)
    F3 = torch.rand(B, pcn_channels[2], 16, 16)
    F4 = torch.rand(B, pcn_channels[3], 8, 8)
    x_t = torch.rand(B, 1, 256, 256)
    t = torch.tensor([10, 500])

    print("\n" + "=" * 50)
    print("🧪 TEST 1: DN CÓ BẬT OCR (mặc định, khớp full paper)")
    print("=" * 50)
    dn_with_ocr = DenoisingNetwork(pcn_channels=pcn_channels, fuse_channels=256, use_ocr=True)
    out_with_ocr = dn_with_ocr(x_t, t, [F1, F2, F3, F4])
    print(f"✅ Output shape: {out_with_ocr.shape}  (kỳ vọng: [B, 1, 256, 256])")

    print("\n" + "=" * 50)
    print("🧪 TEST 2: DN TẮT OCR (dùng cho ablation, Table 4)")
    print("=" * 50)
    dn_without_ocr = DenoisingNetwork(pcn_channels=pcn_channels, fuse_channels=256, use_ocr=False)
    out_without_ocr = dn_without_ocr(x_t, t, [F1, F2, F3, F4])
    print(f"✅ Output shape: {out_without_ocr.shape}  (kỳ vọng: [B, 1, 256, 256])")

    print("\n" + "=" * 50)
    print("🧪 TEST 3: LOSS FUNCTIONS")
    print("=" * 50)
    M_a = (torch.rand(B, 1, 256, 256) > 0.5).float()
    M_v = (torch.rand(B, 1, 256, 256) > 0.5).float()
    bce = WeightedBCELoss()(out_with_ocr, M_a)
    iou = WeightedIoULoss()(out_with_ocr, M_a)
    edge = BoundaryAwareEdgeLoss()(out_with_ocr, M_a, M_v)
    print(f"✅ BCE Loss  : {bce.item():.4f}")
    print(f"✅ IoU Loss  : {iou.item():.4f}")
    print(f"✅ Edge Loss : {edge.item():.4f}")
    print("=" * 50)