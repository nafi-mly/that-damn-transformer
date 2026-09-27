import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


dim   = 16
c_rms = RMSNorm(dim)
try:
    n_rms = nn.RMSNorm(dim)
    n_rms.weight.data = c_rms.weight.data.clone()

    x = torch.randn(2, 4, dim)

    y_c = c_rms(x)
    y_n = n_rms(x)

    print("RMSNorm matches native:", torch.allclose(y_c, y_n, atol=1e-5))

except AttributeError:
    x = torch.randn(2, 4, dim)
    
    y_c = c_rms(x)
    print(y_c.shape)