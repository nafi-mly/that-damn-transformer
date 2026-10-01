import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import math

class _CopyToModelParallelRegion(torch.autograd.Function):
    """
    Forward: Pass the input through as-is.
    Backward: Sum (AllReduce) the gradients across all TP ranks.
    """
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    Forward: Sum (AllReduce) the inputs across all TP ranks.
    Backward: Pass the gradient through as-is.
    """
    @staticmethod
    def forward(ctx, input_):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(input_, op=dist.ReduceOp.SUM)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def copy_to_tensor_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)

def reduce_from_tensor_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)



class ColumnParallelLinear(nn.Module):
    """
    Splits the weight matrix vertically along columns (output dimension).
    Weight shape: [in_features, out_features // world_size]
    """
    def __init__(self, in_features: int, out_features: int, rank: int, world_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size

        assert out_features % world_size == 0, f"Out features ({out_features}) must be divisible by world_size ({world_size})"
        self.out_features_per_partition = out_features // world_size

        # Allocate sliced weight tensor
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_tensor_parallel_region(x)
        output_parallel = nn.functional.linear(input_parallel, self.weight, self.bias)
        return output_parallel

class RowParallelLinear(nn.Module):
    """
    Splits the weight matrix horizontally along rows (input dimension).
    Weight shape: [out_features, in_features // world_size]
    """
    def __init__(self, in_features: int, out_features: int, rank: int, world_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size

        assert in_features % world_size == 0, f"In features ({in_features}) must be divisible by world_size ({world_size})"
        self.in_features_per_partition = in_features // world_size

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear layer without bias (bias is added after AllReduce-Sum across ranks)
        output_parallel = nn.functional.linear(x, self.weight)
        
        # Sum outputs across all TP ranks
        output_ = reduce_from_tensor_parallel_region(output_parallel)
        return output_ + self.bias


class TensorParallelMLP(nn.Module):
    """
    Full Megatron-style MLP block:
    ColumnParallelLinear -> Activation -> RowParallelLinear
    """
    def __init__(self, hidden_dim: int, ffn_dim: int, rank: int, world_size: int):
        super().__init__()
        self.col_proj = ColumnParallelLinear(hidden_dim, ffn_dim, rank, world_size)
        self.act = nn.GELU()
        self.row_proj = RowParallelLinear(ffn_dim, hidden_dim, rank, world_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Column Parallel
        h = self.col_proj(x)
        # 2. Activation
        h = self.act(h)
        # 3. Row Parallel + AllReduce
        out = self.row_proj(h)
        return out

class UnsplitMLP(nn.Module):
    """Standard sequential MLP executed on a single device for ground-truth verification."""
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def run_tp_worker(rank: int, world_size: int, hidden_dim: int = 16, ffn_dim: int = 32):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    torch.manual_seed(1337)
    
    # 1. Instantiate Unsplit Reference Model
    ref_mlp = UnsplitMLP(hidden_dim, ffn_dim)
    
    # 2. Instantiate Tensor Parallel Model
    tp_mlp = TensorParallelMLP(hidden_dim, ffn_dim, rank, world_size)

    with torch.no_grad():
        # Slice Column Parallel Weights (fc1) along output dimension
        col_slice_size = ffn_dim // world_size
        start_col = rank * col_slice_size
        end_col = (rank + 1) * col_slice_size

        tp_mlp.col_proj.weight.copy_(ref_mlp.fc1.weight[start_col:end_col, :])
        tp_mlp.col_proj.bias.copy_(ref_mlp.fc1.bias[start_col:end_col])

        row_slice_size = ffn_dim // world_size
        start_row = rank * row_slice_size
        end_row = (rank + 1) * row_slice_size

        tp_mlp.row_proj.weight.copy_(ref_mlp.fc2.weight[:, start_row:end_row])
        # Only set bias once to prevent double-counting during AllReduce (handled in RowParallelLinear)
        tp_mlp.row_proj.bias.copy_(ref_mlp.fc2.bias)

    # 4. Generate Identical Input Batch Across Ranks
    torch.manual_seed(42)
    x = torch.randn(4, hidden_dim)
    # 5. Forward Pass Comparison
    ref_output = ref_mlp(x)
    tp_output = tp_mlp(x)

    # 6. Verify Exact Output Equality Across Ranks
    max_diff = torch.max(torch.abs(ref_output - tp_output)).item()

    if rank == 0:
        print(f"[Master Rank 0] Reference Output Shape: {ref_output.shape}")
        print(f"[Master Rank 0] TP Output Shape:        {tp_output.shape}")
        print(f"[Master Rank 0] Maximum Absolute Error: {max_diff:.8f}")

        if max_diff < 1e-5:
            print("[Master Rank 0] SUCCESS: Tensor Parallelism produces identical outputs to standard single GPU MLP!\n")
        else:
            print("[Master Rank 0] FAILURE: Divergence detected between TP and Reference outputs!\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 2  # Simulate 2 Tensor Parallel ranks
    print("=========================================================")
    print("       TENSOR PARALLELISM (MEGATRON-LM) FROM SCRATCH     ")
    print("=========================================================")
    print(f"Spawning {WORLD_SIZE} TP worker ranks...\n")

    mp.spawn(
        run_tp_worker,
        args=(WORLD_SIZE, 16, 32),
        nprocs=WORLD_SIZE,
        join=True
    )

    print("=========================================================")
    print(" TP Execution Completed Successfully Across All Ranks!   ")
    print("=========================================================")