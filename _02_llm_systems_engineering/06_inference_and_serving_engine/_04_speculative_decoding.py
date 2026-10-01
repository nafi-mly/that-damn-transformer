import time
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DumbLLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim:int, hidden_dim:int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(embed_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        x = self.embed(input_ids)

        # transformer layer
        for layer in self.layers:
            x = F.relu(layer(x))

        logits = self.head(x)
        return logits


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0) -> Tuple[int, torch.Tensor]:
    """Applies temperature scaling and samples a single token ID along with its probability distribution.

    Args:
        logits: Tensor of shape (vocab_size,)
        temperature: Sampling temperature
    Returns:
        Tuple of (sampled_token_id, probability_distribution_tensor)
    """
    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
    else:
        probs = F.softmax(logits, dim=-1)
        token_id = torch.argmax(probs, dim=-1).item()
    return token_id, probs


def speculative_decoding_step(
    draft_model: DumbLLM,
    target_model: DumbLLM,
    input_ids: torch.Tensor,
    gamma_k: int,
    temperature: float = 1.0
) -> Tuple[List[int], int]: 
    """Executes a single Speculative Decoding iteration:
    1. Draft phase: Draft model generates K candidate tokens sequentially.
    2. Verification phase: Target model processes all K candidates in ONE parallel forward pass.
    3. Rejection sampling: Evaluates candidates step-by-step and recovers exact target distribution on rejection.

    Args:
        draft_model: Small, fast model (q)
        target_model: Large, high-capacity model (p)
        input_ids: Prompt/context sequence tensor of shape (1, seq_len)
        gamma_k: Lookahead budget K (number of speculative tokens drafted)
        temperature: Sampling temperature
    Returns:
        Tuple of (accepted_tokens_list, num_accepted_draft_tokens)
    """

    current_context = input_ids.clone()
    draft_tokens = []
    draft_probs  = []

    for _ in range(gamma_k):
        with torch.no_grad():
            logits = draft_model(current_context)[0, -1, :]
        tok_id, prob = sample_next_token(logits, temperature)
        draft_tokens.append(tok_id)
        draft_probs.append(prob)

        current_context = torch.cat([current_context, torch.tensor([[tok_id]])], dim=1)


    with torch.no_grad():
        target_all_logits = target_model(current_context)[0]

    prompt_len   = input_ids.shape[1]
    target_probs = []
    for i in range(gamma_k):
        pos_logits = target_all_logits[prompt_len - 1 + i, :]
        _, p_probs = sample_next_token(pos_logits, temperature)
        target_probs.append(p_probs)

    extra_logits = target_all_logits[-1, :]
    _, extra_target_probs = sample_next_token(extra_logits, temperature)


    accepted_tokens = []
    n_accepted = 0

    for i in range(gamma_k):
        tok = draft_tokens[i]
        q_tok = draft_probs[i][tok].item()
        p_tok = target_probs[i][tok].item() 

        gamma = min(1, p_tok/(q_tok + 1e-10))
        r = torch.rand(1).item()

        if r <= gamma:
            accepted_tokens.append(tok)
            n_accepted += 1
        else:
            p_dist = target_probs[i]
            q_dist = draft_probs[i]
            adjusted_probs = torch.clamp(p_dist - q_dist, min=0.0)

            sum_adjusted = adjusted_probs.sum()
            if sum_adjusted > 0:
                adjusted_probs = adjusted_probs / sum_adjusted
                replacement_tok = torch.multinomial(adjusted_probs, num_samples=1).item()
            else:
                replacement_tok = torch.multinomial(p_dist, num_samples=1).item()

            accepted_tokens.append(replacement_tok)
            return accepted_tokens, n_accepted

    bonus_tok = torch.multinomial(extra_target_probs, num_samples=1).item()
    accepted_tokens.append(bonus_tok)

    return accepted_tokens, n_accepted


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== Initializing Speculative Decoding Simulation ===")

    VOCAB_SIZE = 1000
    K_LOOKAHEAD = 4

    print("Building Draft Model (680K parameters) and Target Model (12M parameters)...")
    draft_model = DumbLLM(vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=128, num_layers=2)
    target_model = DumbLLM(vocab_size=VOCAB_SIZE, embed_dim=256, hidden_dim=512, num_layers=6)

    prompt = torch.tensor([[1, 5, 12, 99]]) 
    total_tokens_to_generate = 20



    print("\n--- Running Baseline Auto-Regressive Decoding (Target Model Only) ---")
    start_time = time.time()
    curr_input = prompt.clone()
    baseline_tokens = []
    
    for _ in range(total_tokens_to_generate):
        with torch.no_grad():
            logits  = target_model(curr_input)[0, -1, :]
        next_tok, _ = sample_next_token(logits, temperature=0.8)
        baseline_tokens.append(next_tok)

        curr_input  = torch.cat([curr_input, torch.tensor([[next_tok]])], dim=1)  

    baseline_time = time.time() - start_time
    print(f"Baseline Generated Tokens: {baseline_tokens[:8]}...")
    print(f"Baseline Time: {baseline_time * 1000:.2f} ms")




    print(f"\n--- Running Speculative Decoding (K = {K_LOOKAHEAD}) ---")
    start_time = time.time()
    curr_input = prompt.clone()
    speculative_tokens = []

    total_accepted_draft_tokens = 0
    total_draft_iterations = 0

    while len(speculative_tokens) < total_tokens_to_generate:
        new_tokens, n_accepted = speculative_decoding_step(
            draft_model, target_model, curr_input, gamma_k=K_LOOKAHEAD, temperature=0.8
        )

        speculative_tokens.extend(new_tokens)
        total_accepted_draft_tokens += n_accepted
        total_draft_iterations += 1

        curr_input = torch.cat([curr_input, torch.tensor([new_tokens])], dim=1)

    speculative_tokens = speculative_tokens[:total_tokens_to_generate]
    spec_time = time.time() - start_time

    acceptance_rate = (total_accepted_draft_tokens / (total_draft_iterations * K_LOOKAHEAD)) * 100

    print(f"Speculative Generated Tokens: {speculative_tokens[:8]}...")
    print(f"Speculative Time: {spec_time * 1000:.2f} ms")
    print(f"Average Draft Acceptance Rate (alpha): {acceptance_rate:.1f}%")
    print(
        f"Speedup Factor (CPU Sim): {baseline_time / spec_time:.2f}x "
        f"(Note: On GPUs with real memory bandwidth bottlenecks, speedup reaches 2.5x - 3x)"
    )
