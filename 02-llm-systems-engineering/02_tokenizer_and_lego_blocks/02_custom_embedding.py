import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


vocab_size, embed_dim = 1000, 64
c_emb = Embedding(vocab_size, embed_dim)
n_emb = nn.Embedding(vocab_size, embed_dim)

n_emb.weight.data = c_emb.weight.data.clone()
input_ids = torch.tensor([[10, 422, 590, 324], [0, 1, 2, 3]], dtype=torch.long)

c_out = c_emb(input_ids)
n_out = n_emb(input_ids)

print(torch.allclose(c_out, n_out))