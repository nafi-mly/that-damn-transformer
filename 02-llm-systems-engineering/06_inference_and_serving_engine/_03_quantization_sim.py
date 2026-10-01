"""
_03_quantization_sim.py

A pure CPU PyTorch implementation of LLM Quantization techniques.
Simulates Asymmetric Per-Tensor INT8, Symmetric Per-Row INT8, Group-Size INT4,
and Outlier-Aware Mixed-Precision Quantization (LLM.int8() paradigm).
Measures memory footprints, Mean Squared Error (MSE), and outlier resilience.
"""

from typing import Tuple
import torch


def report_quantized_memory(label: str, raw_memory_bytes: int, quantized_bytes: int) -> None:
    """Print compressed storage and the resulting compression ratio."""
    compression_ratio = raw_memory_bytes / quantized_bytes if quantized_bytes > 0 else float("inf")
    print(f"[{label}] Quantized size: {quantized_bytes / (1024 ** 2):.2f} MB | Compression: {compression_ratio:.2f}x")


def quantize_asymmetric_per_tensor_int8(x: torch.Tensor) -> Tuple[torch.Tensor, float ,int]:
    """Asymmetric (Affine) INT8 Quantization over the entire tensor.

    Formula:
        S = (max - min) / 255
        Z = round(-min / S)
        q = clamp(round(x / S) + Z, 0, 255)
    """
    x_min, x_max = x.min().item(), x.max().item()
    scale = (x_max - x_min) / 255.0 if x_max != x_min else 1.0
    zero_point = int(round(-x_min / scale))
    zero_point = max(0, min(255, zero_point))

    q_tensor = torch.clamp(torch.round(x / scale) + zero_point, 0, 255).to(torch.uint8)
    return q_tensor, scale, zero_point

def dequantize_asymmetric_per_tensor_int8(q_tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (q_tensor.to(torch.float32) - zero_point) * scale



def quantize_symmetric_per_row_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 Quantization applied row-by-row (Per-Channel).

    Formula:
        S_i = max(|x_i|) / 127
        q_i = clamp(round(x_i / S_i), -128, 127)  (Z = 0 fixed)
    """
    abs_max_per_row = torch.max(torch.abs(x), dim=1, keepdim=True).values
    scales = abs_max_per_row / 127.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    q_tensor = torch.clamp(torch.round(x / scales), -128, 127).to(torch.int8)
    
    return q_tensor, scales

def dequantize_symmetric_per_row_int8(
        q_tensor: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    return q_tensor.to(torch.float32) * scales



def quantize_group_int4(x: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT4 Quantization divided into sub-block groups.

    Args:
        x: Weight matrix of shape (M, N)
        group_size: Sub-block length along the column dimension
    """

    num_rows, num_cols = x.shape
    assert num_cols % group_size == 0, f"Columns ({num_cols}) must be divisible by group_size ({group_size})"

    x_reshaped = x.reshape(-1, group_size)
    abs_max_per_group = torch.max(torch.abs(x_reshaped), dim=1, keepdim=True).values
    scales = abs_max_per_group / 7.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    q_reshaped = torch.clamp(torch.round(x_reshaped / scales), -8, 7).to(torch.int8)
    q_tensor   = q_reshaped.reshape(num_rows, num_cols)
    scales_tensor = scales.reshape(num_rows, num_cols // group_size)

    return q_tensor, scales_tensor

def dequantize_group_int4(q_tensor: torch.Tensor, scales_tensor: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    num_rows, num_cols = q_tensor.shape
    q_reshaped = q_tensor.reshape(-1, group_size)
    scales_reshaped = scales_tensor.reshape(-1, 1)

    x_dequant = q_reshaped.to(torch.float32) * scales_reshaped
    return x_dequant.reshape(num_rows, num_cols)



def quantize_outlier_aware_int8(
    x:torch.Tensor, 
    outlier_threshold: float = 6.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Outlier-Aware Mixed-Precision Quantization (LLM.int8() style).
    Isolates columns with extreme values (> threshold) to remain in FP16/FP32,
    while quantizing normal columns to Symmetric INT8.
    """

    column_max   = torch.max(torch.abs(x), dim=0).values
    outlier_mask = column_max >= outlier_threshold

    x_outliers = x[:, outlier_mask]
    x_normal   = x[:,~outlier_mask]

    q_normal, normal_scales = quantize_symmetric_per_row_int8(x_normal)
    return q_normal, normal_scales, x_outliers, outlier_mask

def dequantize_outlier_aware_int8(
        q_normal: torch.Tensor,
        normal_scales: torch.Tensor,
        x_outliers: torch.Tensor,
        outlier_mask: torch.Tensor
) -> torch.Tensor:
    num_rows, num_cols = q_normal.shape[0], len(outlier_mask)
    x_reconstructed = torch.zeros((num_rows, num_cols), dtype=torch.float32)

    x_normal_dequant = dequantize_symmetric_per_row_int8(q_normal, normal_scales)

    x_reconstructed[:,~outlier_mask] = x_normal_dequant
    x_reconstructed[:, outlier_mask] = x_outliers.to(torch.float32)

    return x_reconstructed



if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== Initializing Quantization Engine Simulation ===\n")

    M_ROWS, N_COLS = 2048, 8192
    # Create realistic floating-point weights centered around 0.0
    fp32_weights = torch.randn(M_ROWS, N_COLS) * 0.5

    raw_memory_bytes = fp32_weights.element_size() * fp32_weights.numel()
    print(f"Base Weight Matrix Shape: ({M_ROWS}, {N_COLS})")
    print(f"FP32 Baseline Size: {raw_memory_bytes / (1024 ** 2):.2f} MB\n")

    # ------------------------------------------------------------------------
    # BENCHMARK 1: Standard Distributions (No Outliers)
    # ------------------------------------------------------------------------
    print("--- BENCHMARK 1: Standard Gaussian Weight Matrix ---")

    # Method A: Asymmetric Per-Tensor INT8
    q_asym, s_asym, z_asym = quantize_asymmetric_per_tensor_int8(fp32_weights)
    deq_asym = dequantize_asymmetric_per_tensor_int8(q_asym, s_asym, z_asym)
    mse_asym = torch.mean((fp32_weights - deq_asym) ** 2).item()
    asym_mem = q_asym.numel() * q_asym.element_size() + 4 + 4
    report_quantized_memory("Asymmetric Per-Tensor INT8", raw_memory_bytes, asym_mem)

    # Method B: Symmetric Per-Row INT8
    q_sym, s_sym = quantize_symmetric_per_row_int8(fp32_weights)
    deq_sym = dequantize_symmetric_per_row_int8(q_sym, s_sym)
    mse_sym = torch.mean((fp32_weights - deq_sym) ** 2).item()
    sym_mem = q_sym.numel() * q_sym.element_size() + s_sym.numel() * s_sym.element_size()
    report_quantized_memory("Symmetric Per-Row INT8", raw_memory_bytes, sym_mem)

    # Method C: Group-Size INT4 (Group Size = 32)
    q_g4, s_g4 = quantize_group_int4(fp32_weights, group_size=32)
    deq_g4 = dequantize_group_int4(q_g4, s_g4, group_size=32)
    mse_g4 = torch.mean((fp32_weights - deq_g4) ** 2).item()
    group4_mem = int(q_g4.numel() * 0.5) + s_g4.numel() * s_g4.element_size()
    report_quantized_memory("Group-Size 32 INT4", raw_memory_bytes, group4_mem)

    print(f"[Asymmetric Per-Tensor INT8] MSE Loss: {mse_asym:.6f}")
    print(f"[Symmetric Per-Row    INT8] MSE Loss: {mse_sym:.6f}")
    print(f"[Group-Size 32       INT4] MSE Loss: {mse_g4:.6f}")

    # ------------------------------------------------------------------------
    # BENCHMARK 2: Outlier Stress Test (Injecting +25.0 Outlier Feature)
    # ------------------------------------------------------------------------
    print("\n--- BENCHMARK 2: Outlier Stress Test (Emergent Outlier Feature) ---")
    outlier_weights = fp32_weights.clone()
    # Inject massive emergent outlier feature in column #10
    outlier_weights[:, 10] += 25.0

    # Per-Tensor INT8 under Outlier
    q_asym_out, s_asym_out, z_asym_out = quantize_asymmetric_per_tensor_int8(outlier_weights)
    deq_asym_out = dequantize_asymmetric_per_tensor_int8(q_asym_out, s_asym_out, z_asym_out)
    mse_asym_out = torch.mean((outlier_weights - deq_asym_out) ** 2).item()
    asym_out_mem = q_asym_out.numel() * q_asym_out.element_size() + 4 + 4
    report_quantized_memory("Per-Tensor INT8 under Outlier", raw_memory_bytes, asym_out_mem)

    # Per-Row INT8 under Outlier
    q_sym_out, s_sym_out = quantize_symmetric_per_row_int8(outlier_weights)
    deq_sym_out = dequantize_symmetric_per_row_int8(q_sym_out, s_sym_out)
    mse_sym_out = torch.mean((outlier_weights - deq_sym_out) ** 2).item()
    sym_out_mem = q_sym_out.numel() * q_sym_out.element_size() + s_sym_out.numel() * s_sym_out.element_size()
    report_quantized_memory("Per-Row INT8 under Outlier", raw_memory_bytes, sym_out_mem)

    # Outlier-Aware Mixed-Precision (LLM.int8())
    q_norm, s_norm, x_outliers, mask = quantize_outlier_aware_int8(
        outlier_weights, outlier_threshold=10.0
    )
    deq_outlier_aware = dequantize_outlier_aware_int8(q_norm, s_norm, x_outliers, mask)
    mse_outlier_aware = torch.mean((outlier_weights - deq_outlier_aware) ** 2).item()
    outlier_aware_mem = q_norm.numel() * q_norm.element_size() + s_norm.numel() * s_norm.element_size() + x_outliers.numel() * x_outliers.element_size()
    report_quantized_memory("LLM.int8() Outlier-Aware", raw_memory_bytes, outlier_aware_mem)

    print(f"[Per-Tensor INT8] MSE Loss under Outlier : {mse_asym_out:.6f} (CRUSHED PRECISION)")
    print(f"[Per-Row    INT8] MSE Loss under Outlier : {mse_sym_out:.6f}")
    print(f"[LLM.int8() Outlier-Aware] MSE Loss      : {mse_outlier_aware:.6f} (PROTECTED PRECISION!)")
    print(f"Isolated Outlier Columns Count           : {mask.sum().item()} / {N_COLS}")