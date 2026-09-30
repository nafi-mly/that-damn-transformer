"""
Phase 3 - Module 2: Sparse Mixture-of-Experts (MoE) & Dynamic Routing

Demonstrates:
1. Top-K Gating Network / Router Mechanics (N experts, K active).
2. Softmax Routing Weight Normalization over Selected Experts.
3. Token Dispatching & Dynamic Expert Aggregation.
4. Auxiliary Load Balancing Loss computation to prevent "Expert Collapse".
5. Active FLOPs vs. Total VRAM Parameter Footprint Analysis.
"""

