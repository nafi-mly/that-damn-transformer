"""
Supervised Fine-Tuning (SFT) with Masked Cross-Entropy Loss
-----------------------------------------------------------
Demonstrates how prompt tokens are masked with label -100 so loss
is only calculated on target/assistant response tokens.
"""
import torch
import torch.nn as nn

def compute_sft_loss(logits, targets, ignore_index=-100):
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

if __name__ == "__main__":
    print("SFT Masked Loss simulation placeholder initialized.")
