import torch
import torch.nn as nn

from _02_rope_from_scratch import RoPEEmbedding
from _01_causal_attention import CausalSelfAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.silu(self.w_gate(x)) * self.w_up(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, max_seq_len: int = 2048):
        super().__init__()
        self.rms_att = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len)
        
        self.rms_mlp = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ffn=d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pre-Norm Attention + Residual Connection
        h = x + self.attn(self.rms_att(x))

        # 2. Pre-Norm SwiGLU FFN + Residual Connection
        out = h + self.ffn(self.rms_mlp(h))
        
        return out


if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads = 2, 8, 64, 4
    d_ffn = int(2 / 3 * 4 * d_model)  # LLaMA standard hidden dimension scaling
    
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ffn=d_ffn)
    
    x = torch.randn(batch_size, seq_len, d_model)
    out = block(x)
    
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
    
    assert out.shape == x.shape, "Shape mismatch in Transformer Block!"
    print("SUCCESS: Full LLaMA Transformer Block constructed and executed flawlessly!")