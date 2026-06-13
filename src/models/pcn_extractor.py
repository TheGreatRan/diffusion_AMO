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

class PCNExtractor(nn.Module):
    def __init__(self, model_name='pvt_v2_b2', pretrained=True, embed_dim=768):
        super().__init__()
        
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(3)
        )
        self.zoe = ZOE(in_channels=1, embed_dim=64, patch_size=4)
        
        print(f"Loading {model_name} from timm...")
        # Bật lại features_only=True để mạng tự động trích xuất [F1, F2, F3, F4] một cách chuẩn xác
        self.pvt = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
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
        self.time_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(256, c), nn.SiLU()) for c in self.feature_channels
        ])
        
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
        
        # 3. Nhúng Token Thời gian (Broadcast Cộng)
        t_emb = self.time_embed(t)
        fused_features = []
        for F, t_proj in zip(features, self.time_projs):
            t_scale = t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
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
    model = PCNExtractor(model_name='pvt_v2_b2', pretrained=False, embed_dim=768)
    
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