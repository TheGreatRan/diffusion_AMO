import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. MODULE: LOCAL EMPHASIS (LE)
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
        F1_size = features[0].shape[2:] # Lấy (Height, Width) của F1 làm mốc
        f_up = []
        
        for i, f in enumerate(features):
            p = self.projs[i](f) # Đưa về cùng số channels
            if i > 0:
                # Phóng to các đặc trưng sâu (F2, F3, F4) lên bằng F1
                p = F.interpolate(p, size=F1_size, mode='bilinear', align_corners=False)
            f_up.append(p)
            
        return f_up

# ==========================================
# 2. MODULE: ADAPTIVE FEATURE GATE (AFG)
# ==========================================
class AdaptiveFeatureGate(nn.Module):
    def __init__(self, channels=256):
        """Cơ chế cổng kiểm soát (Gate) để dung hợp đặc trưng sâu (A2) và nông (F_up1)."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1) # Xuất ra 1 kênh làm tỷ lệ (Gate)

    def forward(self, A2, F_up1):
        """
        A2: Đặc trưng ngữ nghĩa cấp cao (Shape: B, 256, H/4, W/4)
        F_up1: Đặc trưng chi tiết cấp thấp (Shape: B, 256, H/4, W/4)
        """
        concat_feat = torch.cat([A2, F_up1], dim=1)           # (B, 512, H/4, W/4)
        
        # Gate = Sigmoid(Conv(ReLU(Conv(concat(A2, F_up1)))))
        gate = torch.sigmoid(self.conv2(F.relu(self.conv1(concat_feat)))) # (B, 1, H/4, W/4)
        
        # A1 = Gate * A2 + (1 - Gate) * F_up1
        A1 = gate * A2 + (1.0 - gate) * F_up1                 # (B, 256, H/4, W/4)
        return A1

# ==========================================
# 3. MODULE: DENOISING NETWORK (DN - MẠNG CHÍNH)
# ==========================================
class DenoisingNetwork(nn.Module):
    def __init__(self, pcn_channels=[256, 512, 1024, 2048], fuse_channels=256):
        super().__init__()
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
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # Xuống H/2
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # Xuống H/4 (Bằng với A1)
        
        # Nơi Điều kiện (A1) chèn vào mạng khử nhiễu
        self.bottleneck = nn.Conv2d(256 + fuse_channels, 256, 3, padding=1)
        
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) # Lên H/2
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # Lên H
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1) # Trả về 1 kênh mask (x_hat_0)

    def forward(self, x_t, t_emb, pcn_features):
        """
        x_t: Mask nhiễu (B, 1, H, W)
        t_emb: Timestep embedding (không gian tùy chỉnh nếu cần)
        pcn_features: [F1, F2, F3, F4]
        """
        # 1. Local Emphasis
        F_up1, F_up2, F_up3, F_up4 = self.le_module(pcn_features)
        
        # 2. Progressive Fusion (Dung hợp dần từ sâu ra nông)
        A4 = F_up4
        A3 = self.conv_A3(torch.cat([A4, F_up3], dim=1))
        A2 = self.conv_A2(torch.cat([A3, F_up2], dim=1))
        
        # 3. Adaptive Feature Gate
        A1 = self.afg_module(A2, F_up1) # Shape: (B, 256, H/4, W/4)
        
        # 4. Lightweight U-Net
        e1 = F.relu(self.enc1(x_t))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))      # Shape: (B, 256, H/4, W/4)
        
        # Conditioning: Ghép A1 vào cổ chai (Bottleneck)
        bottleneck_input = torch.cat([e3, A1], dim=1) # Shape: (B, 512, H/4, W/4)
        b = F.relu(self.bottleneck(bottleneck_input))
        
        d1 = F.relu(self.dec1(b))
        d2 = F.relu(self.dec2(d1))
        x_hat_0 = self.final_conv(d2)   # Shape: (B, 1, H, W) (Chưa qua Sigmoid để tính BCE Loss bằng Logits)
        
        return x_hat_0

# ==========================================
# 4. MODULE: HÀM MẤT MÁT (LOSS FUNCTIONS)
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