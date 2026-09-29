import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

class PhysicalKVPool:
    def __init__(
            self,
            num_blocks: int,
            block_size: int,
            num_heads:  int,
            head_dim:   int,
            dtype: torch.dtype = torch.float32, 
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads  = num_heads
        self.head_dim   = head_dim

        self.k_pool = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim), dtype=dtype
        )
        self.v_pool = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim), dtype=dtype
        )


    def write_kv(
            self,
            physical_block_id: int,
            slot_offset: int,
            k_vec: torch.Tensor,
            v_vec: torch.Tensor,
    ) -> None:
        self.k_pool[physical_block_id, slot_offset] = k_vec
        self.v_pool[physical_block_id, slot_offset] = v_vec

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_tables: Dict[str, List[int]] = {}
        self.ref_counts: Dict[int, int] = {i: 0 for i in range(num_blocks)}

    def allocate_block(self, request_id: str) -> int:
        if not self.free_blocks:
            raise MemoryError("Not enough KV Cache Memory lil bro")

        block_id = self.free_blocks.pop(0)
        self.ref_counts[block_id] = 1

        if request_id not in self.block_tables:
            self.block_tables[request_id] = []
        self.block_tables[request_id].append(block_id)

        return block_id

    def get_or_allocate_slot(
            self, request_id:str, token_pos:int
    ):
        logical_block_idx = token_pos // self.block_size
        slot_offset       = token_pos %  self.block_size

        request_blocks = self.block_tables.get(request_id, [])

        if logical_block_idx >= len(request_blocks):
            physical_block_id = self.allocate_block(request_id)
        else:
            physical_block_id = request_blocks[logical_block_idx]

        return physical_block_id, slot_offset

    def free_request(self, request_id: str) -> None:
        if request_id not in self.block_tables:
            return

        for block_id in self.block_tables[request_id]:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.free_blocks.append(block_id)

        del self.block_tables[request_id]

def paged_attention_kernel(
        query: torch.Tensor,
        request_id: str,
        seq_len: int,
        kv_pool: PhysicalKVPool,
        block_manager: BlockManager,
) -> torch.Tensor:
    block_table = block_manager.block_tables[request_id]
    block_size  = block_manager.block_size

    gathered_k_list = []
    gathered_v_list = []

    for physical_block_id in block_table:
        k_block = kv_pool.k_pool[physical_block_id]
        v_block = kv_pool.v_pool[physical_block_id]

        gathered_k_list.append(k_block)
        gathered_v_list.append(v_block)

    full_k = torch.cat(gathered_k_list, dim=0)
    full_v = torch.cat(gathered_v_list, dim=0)

    valid_k = full_k[:seq_len]
    valid_v = full_v[:seq_len]

    valid_k = valid_k.transpose(0, 1)
    valid_v = valid_v.transpose(0, 1)

    q = query.squeeze(0).unsqueeze(1) if query.ndim == 3 else query

    scores = torch.matmul(q, valid_k.transpose(-2, -1)) / math.sqrt(kv_pool.head_dim)
    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, valid_v)

    return output.transpose(1, 0).squeeze(1)


if __name__ == "__main__":
    print("=== Initializing PagedAttention Simulation ===")

    NUM_BLOCKS = 10
    BLOCK_SIZE = 4
    NUM_HEADS = 2
    HEAD_DIM = 8

    # 1. Instantiate Physical Pool and Block Manager
    kv_pool = PhysicalKVPool(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
    manager = BlockManager(NUM_BLOCKS, BLOCK_SIZE)

    print(f"Pool created with {NUM_BLOCKS} blocks (Block Size = {BLOCK_SIZE}).")
    print(f"Initial Free Blocks: {manager.free_blocks}\n")

    # 2. Simulate Generation for Request A (Generating 7 Tokens)
    req_a = "request_alpha"
    print(f"--- Simulating Generation for '{req_a}' (7 Tokens) ---")

    for pos in range(7):
        # Generate random dummy Key and Value vectors for this token step
        k_vec = torch.randn(NUM_HEADS, HEAD_DIM)
        v_vec = torch.randn(NUM_HEADS, HEAD_DIM)

        # Map logical position to physical slot using // and %
        phys_block, slot = manager.get_or_allocate_slot(req_a, pos)

        # Write vectors into physical pool
        kv_pool.write_kv(phys_block, slot, k_vec, v_vec)

        print(
            f"Token Pos {pos:2d} | Math: ({pos} // {BLOCK_SIZE} = {pos // BLOCK_SIZE}, {pos} % {BLOCK_SIZE} = {slot}) "
            f"-> Physical Block #{phys_block}, Slot #{slot}"
        )

    print(f"\nBlock Table for '{req_a}': {manager.block_tables[req_a]}")
    print(f"Remaining Free Blocks: {manager.free_blocks}\n")

    # 3. Compute Attention for Token 7 Query
    query = torch.randn(1, NUM_HEADS, HEAD_DIM)
    attn_output = paged_attention_kernel(query, req_a, 7, kv_pool, manager)

    print(f"Attention Output Shape: {attn_output.shape}")
    print("Successfully calculated PagedAttention output across scattered blocks!\n")

    # 4. Clean up Request
    print(f"--- Freeing Request '{req_a}' ---")
    manager.free_request(req_a)
    print(f"Free Blocks after cleanup: {manager.free_blocks}")