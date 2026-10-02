# Flatten for CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB_SIZE = 100
EMBED_DIM = 32
BETA = 0.1
class LLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids):
        return self.linear(self.embedding(input_ids))

# Policy Model (Trainable) and Reference Model (Frozen Copy)
policy_model = LLM(VOCAB_SIZE, EMBED_DIM)
ref_model = LLM(VOCAB_SIZE, EMBED_DIM)

# Freeze the reference model
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3)

# Batch size = 1
# Prompt: "Explain AI" -> Chosen: Good response, Rejected: Bad response
win_ids = torch.tensor([[10, 15, 20, 30, 40]])   # [Prompt + Chosen]
lose_ids = torch.tensor([[10, 15, 20, 80, 90]])  # [Prompt + Rejected]

# Target masks (-100 on prompt tokens 0..2, calculate loss only on completions 3..4)
win_labels = torch.tensor([[-100, -100, -100, 30, 40]])
lose_labels = torch.tensor([[-100, -100, -100, 80, 90]])


def get_batch_logps(logits, labels):
    """Computes total sequence log-probability for non-masked (-100) target tokens."""
    # Shift logits & labels for autoregressive alignment
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    loss_mask = (shift_labels != -100)
    
    gather_labels = shift_labels.clone()
    gather_labels[~loss_mask] = 0
    
    per_token_logps = torch.gather(log_probs, dim=2, index=gather_labels.unsqueeze(-1)).squeeze(-1)
    
    return (per_token_logps * loss_mask).sum(dim=-1)



# 1. Forward pass on Policy Model
policy_win_logits = policy_model(win_ids)
policy_lose_logits = policy_model(lose_ids)

policy_win_logp = get_batch_logps(policy_win_logits, win_labels)
policy_lose_logp = get_batch_logps(policy_lose_logits, lose_labels)

# 2. Forward pass on Frozen Reference Model 
with torch.no_grad():
    ref_win_logits = ref_model(win_ids)
    ref_lose_logits = ref_model(lose_ids)
    
    ref_win_logp = get_batch_logps(ref_win_logits, win_labels)
    ref_lose_logp = get_batch_logps(ref_lose_logits, lose_labels)

# 3. Calculate DPO Log Ratios
policy_logratios = policy_win_logp - policy_lose_logp
ref_logratios = ref_win_logp - ref_lose_logp

# Implicit reward margin
logits = policy_logratios - ref_logratios

# DPO Loss = -log(sigmoid(beta * logits))
loss = -F.logsigmoid(BETA * logits).mean()

print(f"Policy Win LogP:  {policy_win_logp.item():.4f}")
print(f"Policy Lose LogP: {policy_lose_logp.item():.4f}")
print(f"Calculated DPO Loss: {loss.item():.4f}")

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Backward pass successful! Policy updated towards chosen response.")