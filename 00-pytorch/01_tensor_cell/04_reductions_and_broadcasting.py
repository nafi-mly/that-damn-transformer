import torch
import numpy as np
import inspect

def _print(matrix: torch.tensor) -> None:
    frame = inspect.currentframe().f_back
    call_line = inspect.getframeinfo(frame).code_context[0].strip()
    var_name = call_line.split('_print(')[1].split(')')[0].strip()

    array = matrix.detach().cpu().numpy() if hasattr(matrix, 'detach') else np.asarray(matrix)

    print(f"\n{var_name}:")
    print(np.array2string(array, precision=3, suppress_small=True))



# attention is all you need
q = torch.randn(2, 4, 8, 32)
k = torch.randn(2, 4, 8, 32)

scores = q @ k.transpose(-2, -1)
_print(scores)

# statistics
x = torch.randn(2, 4, 8)
mean_collapsed = x.mean(dim=-1)
mean_preserved = x.mean(dim=-1, keepdim=True)

print(mean_collapsed.shape)
print(mean_preserved.shape)

_print(x)
_print(mean_collapsed)
_print(mean_preserved)

variance = torch.tensor([4.0, 16.0, 25.0])
inv_std  = torch.rsqrt(variance)
_print(inv_std)

# rms_norm
x = torch.randn(2, 16, 512)
gamma = torch.ones(512)
eps = 1e-5

rms_norm = x * (x.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt() * gamma
_print(rms_norm)
_print(x)

# bro i am still confused on how to intuitively think about when is it variable method (like x.mean(...)), when is it torch method (like torch.rsqrt(...)?

def forward(self, x):
    input_dtype = x.dtype
    
    x_fp32   = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_norm   = x_fp32 * torch.sqrt(variance + self.eps)

    return (x_norm * self.weight).to(input_dtype) 