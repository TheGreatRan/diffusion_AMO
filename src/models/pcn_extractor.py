import torch
import torch.nn as nn
import math
import timm

class ZOE(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
    
    def forward(self, x_t):
        return self.proj(x_t)

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
    Đúng tinh thần "Time Token Concatenation" (Sun et al. 2025, được paper CondDiff-AMO
    trích dẫn ở Fig. 4): GHÉP (concatenate) time token vào chuỗi patch token, cho TOÀN BỘ
    (patch + time) cùng đi qua 1 lớp self-attention.

    SỬA LỖI QUAN TRỌNG (phát hiện bởi review, đã tự kiểm chứng bằng số trước khi sửa):
    bản trước dùng CROSS-ATTENTION với Key/Value là DUY NHẤT 1 time token. Về mặt toán học,
    softmax trên đúng 1 phần tử LUÔN LUÔN = 1.0 bất kể nội dung Query -- nghĩa là mọi patch
    nhận về CHÍNH XÁC cùng 1 giá trị, không hề "khác nhau tuỳ patch" như tên gọi/comment cũ
    ngụ ý. Đã verify: out(patch_A) == out(patch_B) tuyệt đối (chênh lệch = 0.0000000000) dù
    patch_A, patch_B khác nhau hoàn toàn. Bản cross-attention cũ về bản chất tương đương
    cơ chế cộng bias cũ (broadcast-add), chỉ là qua 1 phép biến đổi phi tuyến phức tạp hơn.

    Bản sửa này GHÉP THẬT: chuỗi self-attention có (N+1) token (N patch + 1 time), nên
    softmax có nhiều hơn 1 lựa chọn -> patch có thể thực sự nhận trọng số khác nhau tuỳ nội
    dung của chính nó và của các patch khác (patch-to-patch interaction cũng được time token
    "điều tiết" gián tiếp qua chung 1 phép self-attention).

    CHI PHÍ: self-attention đầy đủ có độ phức tạp O((N+1)^2). CHỈ áp dụng module này ở các
    stage có N nhỏ (khuyến nghị: F3 N=256, F4 N=64 với ảnh 256x256) -- ở stage 1/2
    (N=4096/1024), chi phí quá lớn, nên PCNExtractor vẫn dùng broadcast-add (time_projs)
    cho các stage đó dù use_time_token_concat=True (xem forward() của PCNExtractor).
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        # num_heads phải chia hết channels; nếu không, lùi về giá trị chia hết gần nhất
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.time_token_proj = nn.Sequential(nn.Linear(256, channels), nn.SiLU())
        self.self_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, F_i: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        F_i   : (B, C, H, W) -- đặc trưng của 1 stage PVT (CHỈ dùng cho stage có N nhỏ)
        t_emb : (B, 256)     -- time embedding dùng chung (từ TimeEmbedding)
        """
        B, C, H, W = F_i.shape
        tokens = F_i.flatten(2).transpose(1, 2)                 # (B, N, C), N = H*W
        time_token = self.time_token_proj(t_emb).unsqueeze(1)   # (B, 1, C)

        seq = torch.cat([time_token, tokens], dim=1)   # (B, N+1, C) -- GHÉP THẬT SỰ
        seq_norm = self.norm(seq)
        # Self-attention trên TOÀN CHUỖI (patch+time cùng tương tác), không phải cross-attn
        # suy biến với 1 K/V như bản cũ -- (N+1) key/value nên softmax không còn trivial.
        attn_out, _ = self.self_attn(seq_norm, seq_norm, seq_norm)
        seq = seq + attn_out  # residual

        tokens_out = seq[:, 1:, :]  # bỏ time token, giữ lại đúng N patch token
        F_i_out = tokens_out.transpose(1, 2).reshape(B, C, H, W)
        return F_i_out

class PCNExtractor(nn.Module):
    def __init__(self, model_name='pvt_v2_b4', pretrained=True, embed_dim=768, use_time_token_concat=False):
        super().__init__()
        self.use_time_token_concat = use_time_token_concat
        
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(3)
        )

        print(f"Loading {model_name} from timm...")

        self.pvt = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        # Lấy linh động số kênh của Stage 1 (ví dụ B2 là 32, B4 là 64)
        stage1_channels = self.pvt.feature_info.channels()[0]
        # FIX: trước đây self.zoe bị khởi tạo 2 lần (1 lần với embed_dim=64 cứng, bị ghi đè
        # ngay bởi lần thứ 2 với stage1_channels động) -- thừa, dọn lại chỉ còn 1 lần đúng.
        self.zoe = ZOE(in_channels=1, embed_dim=stage1_channels, patch_size=4)

        # ==========================================
        # KỸ THUẬT QUÉT ĐỘNG VÀ GẮN ỐNG TIÊM (HOOK)
        # ==========================================
        embed_layer = None
        # Quét qua toàn bộ các lớp con của PVTv2
        for module in self.pvt.modules():
            # Bắt đúng bản chất Class thay vì dựa vào tên biến (variable name)
            if module.__class__.__name__ in ['OverlapPatchEmbed', 'PatchEmbed']:
                embed_layer = module
                break # Chỉ lấy lớp Patch Embed ĐẦU TIÊN (Tương ứng với Stage 1)
                
        if embed_layer is not None:
            # Gắn Hook: Mỗi khi lớp này chạy xong, tự động gọi hàm _zoe_injection_hook
            embed_layer.register_forward_hook(self._zoe_injection_hook)
        else:
            raise AttributeError("Không tìm thấy lớp Patch Embedding trong kiến trúc timm")
        
        self.feature_channels = self.pvt.feature_info.channels()
        self.time_embed = TimeEmbedding(embed_dim=256)

        # Cơ chế CŨ: broadcast-add bias theo kênh (nhanh, rẻ) -- LUÔN tạo đủ cho cả 4 tầng,
        # vì tầng 1,2 (N=4096/1024 với ảnh 256x256) vẫn dùng cơ chế này ngay cả khi
        # use_time_token_concat=True, do self-attention đầy đủ ở đó quá tốn kém (O(N^2)).
        self.time_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(256, c), nn.SiLU()) for c in self.feature_channels
        ])
        # Cơ chế MỚI (Time Token Concatenation thật sự, xem class ở trên): CHỈ áp dụng cho
        # 2 tầng sâu nhất (index 2, 3 -- tương ứng N=256, 64 token với ảnh 256x256), nơi
        # self-attention đầy đủ O((N+1)^2) vẫn rẻ. Tầng 1, 2 vẫn dùng time_projs phía trên.
        if self.use_time_token_concat:
            self.TTC_STAGE_INDICES = [2, 3]  # chỉ số các tầng dùng cơ chế mới (0-indexed)
            self.time_token_layers = nn.ModuleDict({
                str(i): TimeTokenConcatenation(channels=self.feature_channels[i])
                for i in self.TTC_STAGE_INDICES
            })

        self.hf_proj = nn.Conv2d(self.feature_channels[-1], embed_dim, kernel_size=1)
        
        # Biến trạng thái lưu trữ nhiễu tạm thời cho mỗi batch
        self._current_zoe = None

    def _zoe_injection_hook(self, module, inputs, outputs):
        """Hàm tự động bơm nhiễu x_t vào đầu ra của khối Patch Embed"""
        # Xử lý linh hoạt việc timm có thể trả về Tensor hoặc Tuple(Tensor, H, W)
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        
        zoe_emb = self._current_zoe # Shape chuẩn: (B, C, H, W)
        C = zoe_emb.shape[1]
        
        # Cộng nhiễu an toàn bất chấp định dạng shape mà timm đang dùng
        if x.dim() == 4:
            if x.shape[1] == C:    # (B, C, H, W)
                x = x + zoe_emb
            elif x.shape[-1] == C: # (B, H, W, C)
                x = x + zoe_emb.permute(0, 2, 3, 1)
        elif x.dim() == 3:         # (B, N, C) - Dạng Token chuỗi
            x = x + zoe_emb.flatten(2).transpose(1, 2)
            
        if isinstance(outputs, tuple):
            return (x, *outputs[1:])
        else:
            return x

    def forward(self, I: torch.Tensor, M_v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, use_hf: bool = False):
        X_input = self.conv_c(torch.cat([I, M_v], dim=1))
        
        # 1. Bơm "thuốc nhiễu" ZOE vào biến trạng thái
        self._current_zoe = self.zoe(x_t)
        
        # 2. Chạy mạng PVTv2 như bình thường.
        # Khi đi qua tầng Stage 1, Hook sẽ tự động tiêm ZOE vào.
        # features_only=True sẽ tự động nhả ra mảng [F1, F2, F3, F4] chuẩn xịn
        features = self.pvt(X_input)
        
        # 3. Nhúng Token Thời gian
        t_emb = self.time_embed(t)
        fused_features = []
        for i, F in enumerate(features):
            if self.use_time_token_concat and i in self.TTC_STAGE_INDICES:
                # Cơ chế MỚI (chỉ tầng 3,4 -- N nhỏ, self-attention đầy đủ vẫn rẻ):
                # ghép token thật (concatenation) + self-attention trên toàn chuỗi.
                fused_features.append(self.time_token_layers[str(i)](F, t_emb))
            else:
                # Cơ chế CŨ (tầng 1,2 luôn dùng cái này; tầng 3,4 dùng khi
                # use_time_token_concat=False): broadcast cộng bias cố định theo kênh.
                t_scale = self.time_projs[i](t_emb).unsqueeze(-1).unsqueeze(-1)
                fused_features.append(F + t_scale)
            
        # 4. Điều phối đầu ra cho hệ thống
        if use_hf:
            hf_features = self.hf_proj(fused_features[-1])
            B_hf, C_hf, H_hf, W_hf = hf_features.shape
            return hf_features.view(B_hf, C_hf, -1).permute(0, 2, 1) # Tensor Token cho U-Net
        else:
            return fused_features # List 4 tầng cho Custom DN

# ==========================================
# KHỐI KIỂM THỬ ĐA ĐẦU RA (UNIT TEST ĐỘNG CƠ KÉP)
# ==========================================
if __name__ == "__main__":
    print("🚀 Khởi động Unit Test cho PCN Extractor (PVTv2)...")
    
    # Khởi tạo mô hình
    model = PCNExtractor(model_name='pvt_v2_b4', pretrained=False, embed_dim=768)
    
    # Giả lập Dữ liệu
    I = torch.rand(2, 3, 256, 256)
    M_v = torch.rand(2, 1, 256, 256)
    x_t = torch.rand(2, 1, 256, 256)
    t = torch.tensor([10, 500])
    
    print("\n" + "="*50)
    print("🧪 TEST 1: CHẾ ĐỘ CUSTOM DN (Tác giả gốc)")
    print("="*50)
    pyramid_feats = model(I, M_v, x_t, t, use_hf=False)
    for i, f in enumerate(pyramid_feats):
        print(f"✅ Khối F{i+1} shape: {f.shape}")
        
    print("\n" + "="*50)
    print("🧪 TEST 2: CHẾ ĐỘ HUGGINGFACE U-NET")
    print("="*50)
    hf_tokens = model(I, M_v, x_t, t, use_hf=True)
    print(f"✅ Tensor Token đầu ra: {hf_tokens.shape}")
    print("="*50)