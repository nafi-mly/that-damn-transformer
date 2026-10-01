import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# ==========================================
# 1. SYNTHETIC DATASET
# ==========================================

class SyntheticDataset(Dataset):
    """
    A simple regression dataset where Y = X @ W_true + b_true + noise.
    """
    def __init__(self, num_samples: int = 1000, input_dim: int = 16):
        torch.manual_seed(42)
        self.X = torch.randn(num_samples, input_dim)
        # Ground truth weight vector
        self.W_true = torch.linspace(1.0, 3.0, steps=input_dim).unsqueeze(1)
        self.y = torch.matmul(self.X, self.W_true) + 0.1 * torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 2. TOY MODEL ARCHITECTURE
# ==========================================

class ToyModel(nn.Module):
    """Simple 2-layer MLP for multi-process training."""
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==========================================
# 3. DISTRIBUTED WORKER PROCESS
# ==========================================

def setup_process_group(rank: int, world_size: int, backend: str = "gloo"):
    """
    Initializes the distributed process group.
    'gloo' works across both CPU and GPU; 'nccl' is optimized for CUDA GPUs.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )


def cleanup_process_group():
    """Destroys the process group and releases communication resources."""
    dist.destroy_process_group()


def manual_gradient_all_reduce(model: nn.Module, world_size: int):
    """
    MANUAL RING-ALLREDUCE DEMONSTRATION:
    Instead of relying on PyTorch DDP wrapper magic, this function explicitly 
    loops through each layer's parameter gradient and calls dist.all_reduce() 
    to sum and average the gradients across all ranks.
    """
    for param in model.parameters():
        if param.grad is not None:
            # Step 1: In-place Sum across all processes in the world
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            # Step 2: Divide by total world size to obtain average gradient (bar_g)
            param.grad.data /= world_size


def run_worker_process(rank: int, world_size: int, epochs: int = 5, use_manual_allreduce: bool = False):
    """
    Target entrypoint executed by every spawned process.
    """
    # 1. Initialize process group
    # Automatically fallback to gloo if CUDA GPUs are not available
    backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() >= world_size else "gloo"
    setup_process_group(rank, world_size, backend=backend)

    # Determine device target
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"\n[Master Rank 0] Process group initialized using backend='{backend}' with world_size={world_size}.")
        print(f"[Master Rank 0] Manual Gradient Synchronization: {use_manual_allreduce}\n")

    # 2. Prepare Dataset and Distributed Sampler
    dataset = SyntheticDataset(num_samples=1200, input_dim=16)
    
    # DistributedSampler assigns disjoint sample indices to each rank
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )

    batch_size = 32  # Local mini-batch size per rank
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    # 3. Instantiate Model and Optimizer
    model = ToyModel(input_dim=16, hidden_dim=32, output_dim=1).to(device)

    # Synchronize initial model parameters across all ranks so everyone starts identical
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    if not use_manual_allreduce:
        # Standard Production Approach: Wrap model in PyTorch's DDP
        if backend == "nccl":
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        else:
            model = nn.parallel.DistributedDataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        # CRITICAL: set_epoch guarantees different shuffle seeds across epochs while maintaining rank sync
        sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        samples_processed = 0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            
            # Forward Pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward Pass
            loss.backward()

            # Gradient Synchronization
            if use_manual_allreduce:
                # Manually invoke Ring-AllReduce across ranks
                manual_gradient_all_reduce(model, world_size)
            # (Note: If using DDP wrapper, backward() automatically triggers AllReduce in background)

            # Optimizer Step (executed on synchronized gradients)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            samples_processed += X_batch.size(0)

        avg_loss = total_loss / samples_processed

        # Log epoch progress from Rank 0 only to avoid cluttered stdout
        if rank == 0:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Rank 0 Local Batch Loss: {avg_loss:.4f}")

    cleanup_process_group()


# ==========================================
# 4. MULTI-PROCESS SPAWNER ENGINE
# ==========================================

if __name__ == "__main__":
    WORLD_SIZE = 4  # Spawn 4 distinct OS worker processes
    EPOCHS = 5

    print("=========================================================")
    print("      DISTRIBUTED DATA PARALLEL (DDP) FROM SCRATCH      ")
    print("=========================================================")
    print(f"Spawning {WORLD_SIZE} independent processes via torch.multiprocessing...\n")

    # 1. Run training using PyTorch's native nn.parallel.DistributedDataParallel wrapper
    print("--- RUN 1: Standard PyTorch DDP Engine ---")
    mp.spawn(
        run_worker_process,
        args=(WORLD_SIZE, EPOCHS, False),
        nprocs=WORLD_SIZE,
        join=True
    )

    print("\n---------------------------------------------------------")
    print("--- RUN 2: Manual dist.all_reduce() Gradient Engine ---")
    print("---------------------------------------------------------")
    # 2. Run training using explicit manual AllReduce gradient aggregation
    mp.spawn(
        run_worker_process,
        args=(WORLD_SIZE, EPOCHS, True),
        nprocs=WORLD_SIZE,
        join=True
    )

    print("\n=========================================================")
    print(" DDP Execution Completed Successfully Across All Ranks!  ")
    print("=========================================================")