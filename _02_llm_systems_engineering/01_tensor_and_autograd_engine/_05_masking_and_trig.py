import torch
import numpy as np
import inspect

def _print(matrix: torch.tensor) -> None:
    frame = inspect.currentframe().f_back
    call_line = inspect.getframeinfo(frame).code_context[0].strip()
    var_name = call_line.split('_print(')[1].split(')')[0].strip()

    array = matrix.detach().cpu().numpy() if hasattr(matrix, 'detach') else np.asarray(matrix)

    print(f"\n{var_name}:")
    print(np.array2string(array, precision=3, suppress_small=True))


seq_len = 4
ones = torch.ones(seq_len, seq_len) 
causal_mask =torch.tril(ones) 
_print(causal_mask)

g = torch.Generator().manual_seed(42)
scores = torch.randn(seq_len, seq_len, generator=g)
masked_scores = scores.masked_fill(causal_mask == 0, float('-inf'))
_print(masked_scores)

attn_weights = masked_scores.softmax(dim=-1)
_print(attn_weights)


one = torch.ones(seq_len, seq_len) # we build seq_len x seq_len matrix and fill everything with ones
causal_mask = torch.tril(ones)     # we mask it to be triangle_low_matrix

scores = torch.randn(seq_len, seq_len) # suppose that this is the Q @ K.transpose(0, 1) / torch.sqrt(dim)
masked_scores = scores.masked_fill(causal_mask==0, float('-inf')).softmax(dim=-1) # causal mask operation and softmax! ready to be dot producted with V
_print(masked_scores)


positions = torch.arange(0, seq_len)
freqs = torch.tensor([1.0, 0.1, 0.01])
freqs_grid = torch.outer(positions, freqs)
_print(freqs_grid)


seq_len = 8
raw_scores = torch.randn(seq_len, seq_len)
mask = torch.tril(torch.ones(seq_len, seq_len))

causal_scores = raw_scores.masked_fill(mask==0, float('-inf'))
probs = torch.softmax(causal_scores, dim=-1)

_print(raw_scores)
_print(probs) # it really is intuitive because the longer the context, the more we need to divide/stream our attention to another (previous) tokens
_print(torch.sum(probs[-1]))