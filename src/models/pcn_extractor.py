import math
import timm
import torch
import torch.nn as nn


class ZOE(nn.Module):
    """
    Zero Overlapping Embedding: Tiêm x_t vào Stage 1 của PVT qua conv không chồng lấn.
    
    CẢI TIẾN:
    1. Loại bỏ hoàn toàn InstanceNorm2d để bảo toàn thông tin mật độ che phủ toàn cục (mean) của mask x_t.
    2. Sử dụng GroupNorm(1, embed_dim) (tương đương LayerNorm trên channel) kết hợp với
       hệ số scale học được (self.gate) khởi tạo ở mức 0.1 (Zero/Small-Init paradigm).
    Điều này giữ cho tín hiệu ZOE không làm sốc activation của PVTv2 ImageNet pretrained,
    đồng thời cho phép gradient tự do khuếch đại hoặc thu nhỏ biên độ x_t trong quá trình huấn luyện.
    """
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = nn.GroupNorm(1, embed_dim)
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x_t):
        feat = self.proj(x_t)
        feat = self.norm(feat)
        return feat * self.gate


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class TimeTokenConcatenation(nn.Module):
    """
    Khối Time Token Concatenation chuẩn Transformer Block:
    - Nhúng time token t_emb thành 1 token có độ dài 1 (B, 1, C).
    - Ghép nối trực tiếp vào chuỗi patch tokens: [time_token, patch_tokens] có độ dài (N + 1).
    - Cho toàn bộ chuỗi đi qua Pre-LayerNorm Self-Attention đầy đủ và FFN 2 tầng.
    - Tách bỏ time token ở đầu ra, giữ lại N patch tokens đã được điều biến thời gian sâu.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.time_token_proj = nn.Sequential(
            nn.Linear(256, channels),
            nn.SiLU()
        )
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels)
        )

    def forward(self, F_i: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F_i.shape
        tokens = F_i.flatten(2).transpose(1, 2)  # (B, N, C)
        time_token = self.time_token_proj(t_emb).unsqueeze(1)  # (B, 1, C)

        seq = torch.cat([time_token, tokens], dim=1)  # (B, N + 1, C)

        # Self-Attention Branch với Residual
        seq_norm = self.norm1(seq)
        attn_out, _ = self.self_attn(seq_norm, seq_norm, seq_norm)
        seq = seq + attn_out

        # FFN Branch với Residual
        seq = seq + self.ffn(self.norm2(seq))

        tokens_out = seq[:, 1:, :]  # Loại bỏ time token, lấy lại N patch tokens
        F_i_out = tokens_out.transpose(1, 2).reshape(B, C, H, W)
        return F_i_out


class PCNExtractor(nn.Module):
    def __init__(self, model_name='pvt_v2_b4', pretrained=True, embed_dim=768, use_time_token_concat=True):
        """
        Args:
            model_name (str): Tên backbone PVTv2 từ timm.
            pretrained (bool): Sử dụng pretrained weights ImageNet.
            embed_dim (int): Chiều kênh chiếu nếu dùng HF U-Net.
            use_time_token_concat (bool): MẶC ĐỊNH LÀ TRUE. Kích hoạt ghép time token cho F3, F4.
        """
        super().__init__()
        self.use_time_token_concat = use_time_token_concat

        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(3)
        )

        print(f"Loading {model_name} from timm (pretrained={pretrained})...")
        self.pvt = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.feature_channels = self.pvt.feature_info.channels()

        stage1_channels = self.feature_channels[0]
        self.zoe = ZOE(in_channels=1, embed_dim=stage1_channels, patch_size=4)

        # Bắt lớp Patch Embedding đầu tiên để đăng ký forward hook
        embed_layer = None
        for module in self.pvt.modules():
            if module.__class__.__name__ in ['OverlapPatchEmbed', 'PatchEmbed']:
                embed_layer = module
                break

        if embed_layer is not None:
            embed_layer.register_forward_hook(self._zoe_injection_hook)
        else:
            raise AttributeError("Không tìm thấy lớp Patch Embedding trong kiến trúc timm")

        self.time_embed = TimeEmbedding(embed_dim=256)

        # Broadcast-add bias dự phòng cho F1, F2 (hoặc toàn bộ nếu use_time_token_concat=False)
        self.time_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(256, c), nn.SiLU()) for c in self.feature_channels
        ])

        # Cơ chế ghép token thật cho F3 (16x16=256 tokens) và F4 (8x8=64 tokens)
        self.TTC_STAGE_INDICES = [2, 3]
        if self.use_time_token_concat:
            self.time_token_layers = nn.ModuleDict({
                str(i): TimeTokenConcatenation(channels=self.feature_channels[i])
                for i in self.TTC_STAGE_INDICES
            })

        self.hf_proj = nn.Conv2d(self.feature_channels[-1], embed_dim, kernel_size=1)
        self._current_zoe = None

    def _zoe_injection_hook(self, module, inputs, outputs):
        """Bơm đặc trưng ZOE(x_t) vào ngay sau patch embedding của Stage 1."""
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        zoe_emb = self._current_zoe
        C = zoe_emb.shape[1]

        if x.dim() == 4:
            if x.shape[1] == C:
                x = x + zoe_emb
            elif x.shape[-1] == C:
                x = x + zoe_emb.permute(0, 2, 3, 1)
        elif x.dim() == 3:
            x = x + zoe_emb.flatten(2).transpose(1, 2)

        if isinstance(outputs, tuple):
            return (x, *outputs[1:])
        return x

    def forward(self, I: torch.Tensor, M_v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, use_hf: bool = False):
        X_input = self.conv_c(torch.cat([I, M_v], dim=1))

        # 1. Trích xuất đặc trưng ZOE từ mask nhiễu x_t
        self._current_zoe = self.zoe(x_t)

        # 2. Chạy backbone PVT (hook sẽ tự động tiêm ZOE vào Stage 1)
        features = self.pvt(X_input)

        # 3. Điều biến thời gian (Time Conditioning)
        t_emb = self.time_embed(t)
        fused_features = []
        for i, F in enumerate(features):
            if self.use_time_token_concat and i in self.TTC_STAGE_INDICES:
                fused_features.append(self.time_token_layers[str(i)](F, t_emb))
            else:
                t_scale = self.time_projs[i](t_emb).unsqueeze(-1).unsqueeze(-1)
                fused_features.append(F + t_scale)

        if use_hf:
            hf_features = self.hf_proj(fused_features[-1])
            B_hf, C_hf, H_hf, W_hf = hf_features.shape
            return hf_features.view(B_hf, C_hf, -1).permute(0, 2, 1)
        else:
            return fused_features


if __name__ == "__main__":
    print("🚀 Kiểm thử PCNExtractor với ZOE cải tiến & TimeTokenConcatenation...")
    pcn = PCNExtractor(model_name='pvt_v2_b4', pretrained=False, use_time_token_concat=True)
    
    dummy_I = torch.randn(2, 3, 256, 256)
    dummy_Mv = torch.randn(2, 1, 256, 256)
    dummy_xt = torch.randn(2, 1, 256, 256)
    dummy_t = torch.tensor([100, 500])

    feats = pcn(dummy_I, dummy_Mv, dummy_xt, dummy_t)
    for idx, f in enumerate(feats):
        print(f"✅ Tầng F{idx+1} shape: {f.shape}")