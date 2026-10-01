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



# 1. Packing Real Pairs into Complex Numbers
real_tensor = torch.tensor([
    [1.0, 0.0],  # Vector along Real axis
    [0.0, 1.0],  # Vector along Imaginary axis
    [1.0, 1.0],  # 45-degree vector
    [2.0, 3.0]
])

complex_vec = torch.view_as_complex(real_tensor)
_print(complex_vec)

angles = (torch.pi / 2) * torch.ones(4)
magnitudes = torch.ones_like(angles)

rotators = torch.polar(magnitudes, angles)
_print(rotators)

rotated_complex = complex_vec * rotators
rotated_real = torch.view_as_real(rotated_complex)

_print(rotated_real.round())



q = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]]
])  # Shape: [1, 2, 4]

print("Original q shape:", q.shape)

q_paired = q.view(1, 2, 2, 2)

q_complex = torch.view_as_complex(q_paired)
print("q_complex shape :", q_complex.shape)

theta = torch.tensor([
    [[0.0, 0.0],
     [torch.pi, torch.pi]]
])  # Shape: [1, 2, 2]

rotator = torch.polar(torch.ones_like(theta), theta)

q_rotated_complex = q_complex * rotator

q_rotated_paired = torch.view_as_real(q_rotated_complex)

q_rotated = q_rotated_paired.reshape(1, 2, 4)

print("\nq_rotated shape:", q_rotated.shape)
print("q_rotated values:\n", q_rotated.round())



def precompute_freqs_cis(dim:int, seq_len:int, tetha: float = 10000.0) -> torch.Tensor :
    freqs = 1.0 / (tetha ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs) 

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_pairs)

    freqs_cis = freqs_cis.view(1, x.shape[1], 1, -1) 
    x_rotated_complex = x_complex * freqs_cis
    
    x_rotated_pairs = torch.view_as_real(x_rotated_complex)
    
    x_out = x_rotated_pairs.reshape(*x.shape)

    return x_out.type_as(x)

# Parameters
batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 64

# Generate random input vector x
x = torch.randn(batch_size, seq_len, num_heads, head_dim)

# Precompute rotation frequencies
freqs_cis = precompute_freqs_cis(dim=head_dim, seq_len=seq_len)

# Apply RoPE
x_rotated = apply_rope(x, freqs_cis)

# Verification 1: Shape check
assert x_rotated.shape == x.shape, f"Expected shape {x.shape}, got {x_rotated.shape}"

# Verification 2: Magnitude check (Geometric length MUST be invariant!)
orig_norm = torch.norm(x, dim=-1)
rotated_norm = torch.norm(x_rotated, dim=-1)
max_diff = (orig_norm - rotated_norm).abs().max().item()

print(f"Shape check passed: {x_rotated.shape}")
print(f"Max length deviation after rotation: {max_diff:.8f}")

if max_diff < 1e-5:
    print("SUCCESS: RoPE accurately rotated vectors while preserving 100% of semantic magnitude!")