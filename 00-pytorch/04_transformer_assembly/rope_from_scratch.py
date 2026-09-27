import torch
import torch.nn as nn

class RoPEEmbedding(nn.Module):
    def __init__(self, head_dim:int, max_seq_len:int = 2048, theta:float = 10000.0):
        super().__init__()
        self.head_dim = head_dim

        # we arange 0 till head, but jumps twice. Then, we strip it just until head/2, 
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

        # dimension for each dimension slower rotation
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)

        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, q:torch.Tensor, k:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]

        freqs_cis = self.freqs_cis[:seq_len]
        freqs_cis = freqs_cis.view(1, seq_len, 1, -1)

        q_pairs = q.float().reshape(*q.shape[:-1], -1, 2)
        k_pairs = k.float().reshape(*k.shape[:-1], -1, 2)

        q_complex = torch.view_as_complex(q_pairs)
        k_complex = torch.view_as_complex(k_pairs)

        q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
        k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(-2)

        return q_rotated.type_as(q), k_rotated.type_as(k)


if __name__ == "__main__":
    batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 64

    rope = RoPEEmbedding(head_dim=head_dim, max_seq_len=1024)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)

    q_rot, k_rot = rope(q, k)
    print("q shape:", q.shape, "-> q_rot shape:", q_rot.shape)
    print("k shape:", k.shape, "-> k_rot shape:", k_rot.shape)
    
    norm_diff_q = (q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item()
    norm_diff_k = (k.norm(dim=-1) - k_rot.norm(dim=-1)).abs().max().item()
    
    print(f"Max length difference Q: {norm_diff_q:.8f}")
    print(f"Max length difference K: {norm_diff_k:.8f}")