import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x *  torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ffn:int):
        super().__init__()

        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up   = nn.Linear(d_model, d_ffn, bias=False)

        self.w_down = nn.Linear(d_ffn, d_model, bias=False)
        self.silu   = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.w_gate(x))
        up = self.w_up(x)

        hidden = gate * up

        return self.w_down(hidden)

x = torch.randn(2, 4, 16)
c_silu = SiLU()(x)
n_silu = F.silu(x)

swiglu = SwiGLU(d_model=16, d_ffn=64)
out    = swiglu(x)
print(out.shape)