import torch

t1 = torch.randn(3)
t2 = torch.randn(2, 2, 3, 4, 2)
t3 = torch.zeros(3)
print(t3.dtype)
t4 = torch.arange(3, 6)

q = torch.randn(2, 16, 8, 64)
print(q.shape)