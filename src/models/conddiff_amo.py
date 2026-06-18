import torch
import torch.nn as nn
from .pcn_extractor import PCNExtractor
from .denoising_net import DenoisingNetwork

class CondDiffAMO(nn.Module):
    def __init__(self, pcn_model_name='pvt_v2_b2', pretrained=True, fuse_channels=256):
        """
        Class đại diện cho toàn bộ kiến trúc Mạng Khử Nhiễu Có Điều Kiện (CondDiff-AMO).
        Đóng gói hoàn hảo cả PCN và DN vào một thực thể duy nhất.
        """
        super().__init__()
        
        # 1. Khởi tạo Động cơ Trích xuất Điều kiện (PCN)
        self.pcn = PCNExtractor(model_name=pcn_model_name, pretrained=pretrained)
        
        # 2. Khởi tạo Động cơ Khử nhiễu (DN) tự động khớp số kênh với PCN
        self.dn = DenoisingNetwork(
            pcn_channels=self.pcn.feature_channels, 
            fuse_channels=fuse_channels
        )

    def forward(self, I: torch.Tensor, M_v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, use_hf: bool = False):
        """
        Luồng truyền xuôi (Forward Pass) cho quá trình Huấn luyện (Training).
        """
        # Bước 1: PCN trích xuất đặc trưng hình học và nhúng thời gian
        pyramid_features = self.pcn(I, M_v, x_t, t, use_hf=use_hf)
        
        # Bước 2: DN nhận đặc trưng điều kiện để đoán lại nhiễu (hoặc đoán x_0)
        x_hat_0_logits = self.dn(x_t, t, pyramid_features)
        
        return x_hat_0_logits