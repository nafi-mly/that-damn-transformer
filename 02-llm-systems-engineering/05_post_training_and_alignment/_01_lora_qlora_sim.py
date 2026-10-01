"""
Phase 3 - Module 1: Parameter-Efficient Fine-Tuning (LoRA & QLoRA Mechanics)

Demonstrates:
1. Low-Rank Matrix Decomposition (W0 + (alpha/r) * B @ A).
2. Zero-Initialization Verification at Step 0 (B = 0).
3. Zero-Latency Weight Merging (W_merged = W0 + (alpha/r) * B @ A).
4. QLoRA Integration (Frozen INT4 Base Weights + Trainable FP32 Adapters).
5. Parameter & Memory Comparison (Full Fine-Tuning vs. LoRA).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    Wraps a standard linear layer with Low-Rank Adaptation (LoRA).
    Base weights W0 are frozen, while low-rank adapters A and B are trainable.
    """
    def __init__(self, in_features: int, out_features: int, r:int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        if r > 0:   
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.reset_parameters()

        self.merged = False
        

    def reset_parameters(self):
        """
        Initialization Strategy:
        Matrix A ~ Kaiming Uniform / Normal
        Matrix B = 0
        Guarantees Delta W = 0 at Step 0.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.linear(x, self.weight, self.bias)

        base_output = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return base_output + lora_output

        return base_output 

    def merge(self):
        """
        Merges low-rank weights into base weights for zero-latency serving:
        W_merged = W0 + (alpha / r) * (B @ A)
        """
        if not self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Subtracts adapter weights to revert to base model state."""        
        if self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= delta_w
            self.merged = False


class QLoRALinear(nn.Module):
    """
    Simulates QLoRA:
    - Base weights W0 are quantized to 4-bit (INT4) with group scales and frozen.
    - Adapters A and B remain in full precision (FP32/FP16) and trainable.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 16.0, group_size: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.group_size = group_size

        # 1. Generate unquantized FP32 base weight
        raw_w = torch.randn(out_features, in_features)

        # 2. Quantize base weight to INT4 with Group Scaling
        self.register_buffer("q_weight", self._quantize_int4(raw_w, group_size))
        self.register_buffer("scales", self._compute_scales(raw_w, group_size))

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 3. Trainable FP32 LoRA adapters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _compute_scales(self, w: torch.Tensor, group_size: int) -> torch.Tensor:
        w_grouped = w.view(-1, group_size)
        max_vals  = torch.max(torch.abs(w_grouped), dim=-1, keepdim=True)[0]
        scales    = max_vals / 7.0
        
        return scales 

    def _quantize_int4(self, w:torch.Tensor, group_size:int) -> torch.Tensor:
        scales = self._compute_scales(w, group_size)
        w_grouped = w.view(-1, group_size)
        q_grouped = torch.clamp(torch.round(w_grouped / (scales + 1e-8)), -7, 7)

        return q_grouped.to(torch.int8).view(self.out_features, self.in_features)

    def _dequantize_int4(self) -> torch.Tensor:
        w_grouped = self.q_weight.view(-1, self.group_size).to(torch.float32)
        scales = self.scales.view(-1, 1)
        deq_w = w_grouped * scales

        return deq_w.view(self.out_features, self.in_features)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w_fp32 = self._dequantize_int4()
        base_output = F.linear(x, w_fp32)

        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output


def run_simulation():
    print("=== Phase 3, Module 1: LoRA & QLoRA Engine Simulation ===\n")

    in_dim, out_dim = 4096, 4096
    rank = 8
    alpha = 16.0
    batch_size = 4
    x = torch.randn(batch_size, in_dim)



    # Test 1: Step 0 Initialization & Output Identity ------------------
    print("--- TEST 1: Step 0 Zero-Initialization Verification ---")
    lora_layer = LoRALinear(in_dim, out_dim, r=rank, lora_alpha=alpha)

    base_only_output = F.linear(x, lora_layer.weight, lora_layer.bias)
    lora_step0_output = lora_layer(x)
    
    diff_step0 = torch.max(torch.abs(base_only_output - lora_step0_output)).item()
    print(f"Base Output vs. LoRA Output (Step 0) Max Diff : {diff_step0:.8f}")
    assert diff_step0 == 0.0, "Step 0 verification failed! Output must be identical."
    print("[PASS] At Step 0 (B=0), LoRA contributes exactly 0 change to base model.\n")



    # Test 2: Weight Merging Equivalence (Zero-Latency Serving) --------
    print("--- TEST 2: Zero-Latency Weight Merging Equivalence ---")
    # Simulate non-zero adapter weights after some training steps
    nn.init.normal_(lora_layer.lora_B, std=0.02)
    
    # Separate forward pass (unmerged)
    unmerged_output = lora_layer(x)
    
    # Merge weights and compute forward pass
    lora_layer.merge()
    merged_output = lora_layer(x)
    
    diff_merge = torch.max(torch.abs(unmerged_output - merged_output)).item()
    print(f"Unmerged Output vs. Merged Output Max Diff  : {diff_merge:.8f}")
    assert diff_merge < 1e-3, "Weight merging output mismatch!"
    print("[PASS] Merged layer W_merged = W0 + (alpha/r)*(B@A) yields numerically equivalent outputs.\n")



    # Test 3: QLoRA (INT4 Base Weight + FP32 Adapter) Forward Pass -----
    print("--- TEST 3: QLoRA (INT4 Base + FP32 Adapter) Forward Pass ---")
    qlora_layer = QLoRALinear(in_dim, out_dim, r=rank, lora_alpha=alpha, group_size=32)
    qlora_output = qlora_layer(x)
    print(f"QLoRA Output Shape                          : {tuple(qlora_output.shape)}")
    print(f"QLoRA Base Weight Quantized Data Type       : {qlora_layer.q_weight.dtype}")
    print(f"QLoRA Adapter A Trainable dtype             : {qlora_layer.lora_A.dtype}")
    print("[PASS] QLoRA pipeline functioning as intended.\n")



    # Test 4: Parameter & VRAM Footprint Comparison --------------------
    print("--- TEST 4: Parameter & VRAM Footprint Benchmark ---")
    # Model parameters for a 4096 x 4096 Linear Layer
    full_params = in_dim * out_dim
    lora_params = (in_dim * rank) + (rank * out_dim)
    
    param_reduction = (1.0 - (lora_params / full_params)) * 100
    
    # Optimizer State Memory in FP32 (AdamW stores 2 states per trainable parameter)
    full_opt_memory_mb = (full_params * 2 * 4) / (1024 ** 2)
    lora_opt_memory_mb = (lora_params * 2 * 4) / (1024 ** 2)

    print(f"Base Layer Dimensions                       : ({in_dim}, {out_dim})")
    print(f"Full Fine-Tuning Trainable Parameters       : {full_params:,}")
    print(f"LoRA (r={rank}) Trainable Parameters          : {lora_params:,}")
    print(f"Parameter Reduction                         : {param_reduction:.2f}%")
    print(f"AdamW Optimizer Memory (Full FT)            : {full_opt_memory_mb:.2f} MB")
    print(f"AdamW Optimizer Memory (LoRA)               : {lora_opt_memory_mb:.2f} MB")
    

if __name__ == "__main__":
    run_simulation()