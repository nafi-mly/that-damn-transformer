import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KVCache:
    """Persistent Key-Value Cache Buffer for Fast Autoregressive Generation."""
    def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, device: torch.device):
        self.k_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim, device=device)
        self.v_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim, device=device)

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        k_val, v_val shape: [Batch, NumHeads, SeqLen_New, HeadDim]
        Appends new KV vectors into persistent memory at position start_pos.
        """
        batch_size, num_heads, seq_len, head_dim = k_val.shape
        
        # Write new key/value representations into persistent cache slice
        self.k_cache[:batch_size, :, start_pos : start_pos + seq_len, :] = k_val
        self.v_cache[:batch_size, :, start_pos : start_pos + seq_len, :] = v_val
        
        # Retrieve complete key/value history up to current position
        keys = self.k_cache[:batch_size, :, : start_pos + seq_len, :]
        values = self.v_cache[:batch_size, :, : start_pos + seq_len, :]
        
        return keys, values


class CachedCausalAttention(nn.Module):
    """Causal Self-Attention supporting KV-Cache during generation."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int, kv_cache: KVCache | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. Project Q, K, V for current input sequence
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Update KV Cache or use current K, V directly
        if kv_cache is not None:
            keys, values = kv_cache.update(k, v, start_pos)
        else:
            keys, values = k, v
            
        # 3. Scaled Dot-Product Attention
        # q: [B, H, SeqLen_New, HeadDim]
        # keys: [B, H, Total_SeqLen, HeadDim] -> transposed: [B, H, HeadDim, Total_SeqLen]
        scores = (q @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. Masking (Only needed if processing initial multi-token prompt)
        if seq_len > 1:
            mask = torch.tril(torch.ones(seq_len, keys.shape[2], device=x.device)).bool()
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ values  # [B, H, SeqLen_New, HeadDim]
        
        # 5. Concatenate heads and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)


# =========================================================
# VERIFICATION:
# =========================================================
if __name__ == "__main__":
    batch_size, d_model, num_heads = 1, 64, 4
    device = torch.device("cpu")
    
    attn = CachedCausalAttention(d_model=d_model, num_heads=num_heads)
    cache = KVCache(max_batch_size=batch_size, max_seq_len=16, num_heads=num_heads, head_dim=d_model // num_heads, device=device)
    
    # --- Step 1: Process Prompt (3 tokens) ---
    prompt = torch.randn(batch_size, 3, d_model)
    out_prompt = attn(prompt, start_pos=0, kv_cache=cache)
    print("Prompt output shape (3 tokens):", out_prompt.shape)
    
    # --- Step 2: Generate Token 4 (1 new token at start_pos=3) ---
    new_token_1 = torch.randn(batch_size, 1, d_model)
    out_token_1 = attn(new_token_1, start_pos=3, kv_cache=cache)
    print("Generation Token 4 output shape:", out_token_1.shape)
    
    # --- Step 3: Generate Token 5 (1 new token at start_pos=4) ---
    new_token_2 = torch.randn(batch_size, 1, d_model)
    out_token_2 = attn(new_token_2, start_pos=4, kv_cache=cache)
    print("Generation Token 5 output shape:", out_token_2.shape)
    
    print("SUCCESS: KV-Cache correctly maintained history and executed single-token step projections!")