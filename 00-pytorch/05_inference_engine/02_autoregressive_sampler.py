import torch
import torch.nn as nn
import torch.nn.functional as F

def sampling(logits: torch.Tensor, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


if __name__ == "__main__":
    vocab_size = 1000
    batch_size = 1
    
    g = torch.Generator().manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, generator=g)
    
    # 2. Test Greedy Sampling (T=0.0)
    greedy_token = sampling(logits.clone(), temperature=0.0)
    print("Greedy Selected Token ID:", greedy_token.item())
    
    # 3. Test Temperature Sampling (T=0.7, Top-K=50, Top-P=0.9)
    sampled_token = sampling(logits.clone(), temperature=0.7, top_k=100, top_p=0.6)
    print("Top-K/Top-P Sampled Token ID:", sampled_token.item())
    
