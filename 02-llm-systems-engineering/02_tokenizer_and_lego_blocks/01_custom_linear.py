import torch
import torch.nn as nn
import math

class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out

in_dim, out_dim = 16, 32
custom_layer = CustomLinear(in_dim, out_dim, bias=True)
native_layer = nn.Linear(in_dim, out_dim, bias=True)

native_layer.weight.data = custom_layer.weight.data.clone()
native_layer.bias.data   = custom_layer.bias.data.clone()

x = torch.randn(2, 4, 16)

y_custom = custom_layer(x)
y_native = native_layer(x)

print(torch.allclose(y_custom, y_native))