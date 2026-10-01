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



x = torch.randn(5, 8)
xs = x.view(2, 5, 4).float()

latest_token = xs[-1, -1:, :]

# [batch, heads, pastseq, headdim]
past_keys = torch.randn(1, 2, 4, 6) 
new_key   = torch.randn(1, 2, 1, 6)

updated_keys = torch.cat([past_keys, new_key], dim=2)
_print(past_keys)
_print(new_key)
_print(updated_keys)



# we simulate storing new key weight at **ONE LAYER** among all layers on the (last) token
# this code is just for one fucking layer, and inside, we're just dealing with K 
cached_k = torch.randn(2, 8, 10, 64) # we have 8 stored tokens in 512 dimensions on two batches
new_k    = torch.randn(2, 8, 1, 64) # suppose this is our calculation result after inputing one (last) token to W_k 

updated_k = torch.cat([cached_k, new_k], dim=2)
latest_token = updated_k[:, :, -1:, :]
_print(latest_token) # this means we have one last token with 512 dimensions,
                    # but we chunks them into 8 pieces. We chunks them so that it can be 
                    # independently calculated parallelly? how to join all of them? 
                    # we may (forcely) interpret dim 0-64 as syntax, 65-128 as grammar, and so on... 
                    # but we will never know the exact interpretation.
                    # maybe it's meaningless for human, but not for machine.   

joined_token = latest_token.permute(0, 2, 1, 3).contiguous().view(2, 1, 8 * 64)
_print(joined_token); print(joined_token.shape) # we have TWO different last tokens, 
                                               # it means that we compute two different (last) tokens
                                               # for two different people.
                                               # in real life, the batch is way bigger


k_cache = torch.rand(1000, )