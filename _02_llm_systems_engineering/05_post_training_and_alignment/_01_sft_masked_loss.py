import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB_SIZE = 100
EMBED_DIM = 32
IGNORE_INDEX = -100  

class LLM(nn.Module):
    """Minimal autoregressive model to simulate logits generation."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        logits = self.linear(x)
        return logits

model = LLM(VOCAB_SIZE, EMBED_DIM)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# Imagine sequence: "User: Hello! Assistant: Hi there!"
# Token IDs (Length = 8):
input_ids = torch.tensor([[10, 15, 20, 25, 30, 35, 40, 45]])  

labels = input_ids.clone()
labels[:, :5] = IGNORE_INDEX  

print("Input IDs: ", input_ids)
print("Target Labels:", labels)  

# Forward pass
logits = model(input_ids)

# Logits predicts position i+1 from position i
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
loss = loss_fn(
    shift_logits.view(-1, VOCAB_SIZE), 
    shift_labels.view(-1)
)

print(f"\nCalculated SFT Loss: {loss.item():.4f}")

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Backward pass successful! Model parameters updated on assistant tokens only.")