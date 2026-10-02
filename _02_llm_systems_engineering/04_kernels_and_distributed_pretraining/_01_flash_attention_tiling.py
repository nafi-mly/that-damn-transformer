import math
import torch
import torch.nn as nn

def std_attention(Q: torch.Tensor, K: torch.Tensor, V:torch.Tensor, causal: bool = True) -> torch.Tensor:
    """
    Standard PyTorch Multi-Head Attention execution:
    Computes S = Q @ K.T / sqrt(d), applies causal mask, materializes full Softmax(S) in VRAM,
    and returns O = P @ V.
    
    Shapes:
        Q, K, V: [batch_size, num_heads, seq_len, head_dim]
    """

    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output

def flash_attention_tiled(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    causal: bool = True,
    block_size_r: int = 32, 
    block_size_c: int = 32
) -> torch.Tensor:
    """
    FlashAttention simulation using block-by-block Online Softmax tiling.
    
    NEVER materializes the full [N, N] matrix in memory!
    Processes Query blocks (rows) in the outer loop and Key/Value blocks (cols) in the inner loop.
    """

    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    # Output tensor to hold accumulated weighted values
    O = torch.zeros_like(Q)

    num_blocks_r = math.ceil(seq_len / block_size_r)
    num_blocks_c = math.ceil(seq_len / block_size_c)

    for b in range(batch_size):
        for h in range(num_heads):
            q_bh = Q[b, h]  # [N, d]
            k_bh = K[b, h]  # [N, d]
            v_bh = V[b, h]  # [N, d]

            for i in range(num_blocks_r):
                r_start = i * block_size_r
                r_end = min((i + 1) * block_size_r, seq_len)
                q_tile = q_bh[r_start:r_end, :]  # Block Q_i [B_r, d]

                actual_br = r_end - r_start
                m_old = torch.full((actual_br, 1), float("-inf"), device=Q.device)
                d_old = torch.zeros((actual_br, 1), device=Q.device)
                o_tile = torch.zeros((actual_br, head_dim), device=Q.device, dtype=Q.dtype)

                for j in range(num_blocks_c):
                    c_start = j * block_size_c
                    c_end = min((j + 1) * block_size_c, seq_len)

                    if causal and c_start > r_end - 1:
                        continue

                    k_tile = k_bh[c_start:c_end, :]
                    v_tile = v_bh[c_start:c_end, :]

                    s_ij = torch.matmul(q_tile, k_tile.transpose(0, 1)) * scale

                    if causal:
                        row_idx = torch.arange(r_start, r_end, device=Q.device).unsqueeze(1)
                        col_idx = torch.arange(c_start, c_end, device=Q.device).unsqueeze(0)
                        causal_mask = col_idx > row_idx
                        s_ij = s_ij.masked_fill(causal_mask, float("-inf"))

                    m_local, _ = torch.max(s_ij, dim=-1, keepdim=True)
                    m_new = torch.maximum(m_old, m_local)
                    alpha = torch.exp(m_old - m_new)
                    p_ij = torch.exp(s_ij - m_new)

                    d_new = (d_old * alpha) + torch.sum(p_ij, dim=-1, keepdim=True)
                    o_tile = (o_tile * alpha) + torch.matmul(p_ij, v_tile)

                    m_old = m_new
                    d_old = d_new

                O[b, h, r_start:r_end, :] = o_tile / d_old

    return O

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Configuration
    BATCH_SIZE = 2
    NUM_HEADS = 4
    SEQ_LEN = 256     # Sequence length N
    HEAD_DIM = 64     # Head dimension d
    BLOCK_SIZE = 64   # Tile size (B_r = B_c = 64)
    
    print("=========================================================")
    print("        FLASHATTENTION SRAM TILING FROM SCRATCH         ")
    print("=========================================================")
    print(f"Shape Config: Batch={BATCH_SIZE}, Heads={NUM_HEADS}, SeqLen={SEQ_LEN}, HeadDim={HEAD_DIM}")
    print(f"SRAM Tile Block Size: {BLOCK_SIZE}x{BLOCK_SIZE}\n")
    
    # Random Query, Key, Value Tensors
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    
    # 1. Run Unflashed Reference Attention
    print("Computing Standard Unflashed Attention (Materializing N x N)...")
    out_ref = std_attention(Q, K, V, causal=True)
    
    # 2. Run SRAM-Tiled FlashAttention
    print("Computing FlashAttention (Online Softmax SRAM Tiling)...")
    out_flash = flash_attention_tiled(
        Q, K, V, 
        causal=True, 
        block_size_r=BLOCK_SIZE, 
        block_size_c=BLOCK_SIZE
    )
    
    # 3. Measure Max Absolute Difference
    max_diff = torch.max(torch.abs(out_ref - out_flash)).item()
    
    print("\n---------------------------------------------------------")
    print(f"Reference Output Tensor Shape: {out_ref.shape}")
    print(f"FlashAttention Output Shape:   {out_flash.shape}")
    print(f"Maximum Absolute Difference:   {max_diff:.8e}")
    print("---------------------------------------------------------")
    
    if max_diff < 1e-5:
        print(" SUCCESS: FlashAttention Online Softmax tiling produces mathematically")
        print("          identical outputs to standard un-tiled attention!")
    else:
        print(" FAILURE: Output discrepancy detected!")
        
    print("=========================================================")