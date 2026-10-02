import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB_SIZE = 100
EMBED_DIM = 32
BETA = 0.04
GROUP_SIZE = 4 

class LLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids):
        return self.linear(self.embedding(input_ids))

policy_model = LLM(VOCAB_SIZE, EMBED_DIM)
ref_model = LLM(VOCAB_SIZE, EMBED_DIM)

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3)



# Single prompt repeated GROUP_SIZE times
prompt_ids = torch.tensor([[10, 15, 20]]).repeat(GROUP_SIZE, 1) # [G, Prompt_Len]

# Simulated sampled completion tokens for each group member
# Assume target token 99 is the "correct reasoning step"
group_completions = torch.tensor([
    [30, 40, 99], # Response 1 (Contains 99 -> Good)
    [30, 40, 50], # Response 2 (No 99 -> Bad)
    [30, 99, 99], # Response 3 (Two 99s -> Great)
    [12, 14, 16]  # Response 4 (Irrelevant -> Bad)
])

# Full sequence = Prompt + Completion
input_ids = torch.cat([prompt_ids, group_completions], dim=-1)

# Target mask (-100 on prompt tokens)
labels = input_ids.clone()
labels[:, :3] = -100

def compute_rewards(completions):
    """Rule-based reward function: count occurrences of token 99."""
    rewards = (completions == 99).sum(dim=-1).float()
    return rewards

rewards = compute_rewards(group_completions) 

# Group Normalization -> Advantages
mean_r = rewards.mean()
std_r = rewards.std() + 1e-8 
advantages = (rewards - mean_r) / std_r

print("Raw Rewards:      ", rewards.tolist())
print("Group Advantages: ", [round(a, 4) for a in advantages.tolist()])


def get_per_token_logps(logits, labels):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    loss_mask = (shift_labels != -100)
    
    gather_labels = shift_labels.clone()
    gather_labels[~loss_mask] = 0
    
    per_token_logps = torch.gather(log_probs, dim=2, index=gather_labels.unsqueeze(-1)).squeeze(-1)
    return per_token_logps, loss_mask


policy_logits = policy_model(input_ids)
policy_logps, loss_mask = get_per_token_logps(policy_logits, labels)

with torch.no_grad():
    ref_logits = ref_model(input_ids)
    ref_logps, _ = get_per_token_logps(ref_logits, labels)

kl_per_token = torch.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1

advantage_weight = advantages.unsqueeze(-1) # [G, 1]
per_token_loss = -(policy_logps * advantage_weight) + (BETA * kl_per_token)

loss = (per_token_loss * loss_mask).sum(dim=-1).mean()

print(f"\nCalculated GRPO Loss: {loss.item():.4f}")

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Backward pass successful! Policy updated towards high-reward reasoning tokens.")