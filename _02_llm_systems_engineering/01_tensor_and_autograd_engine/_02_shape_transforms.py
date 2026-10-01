import torch
x = torch.randn(2, 4, 6)
x_heads = x.view(2, 4, 2, 3)

# print(x)
# print(x_heads)

x_perm = x_heads.permute(0, 3, 2, 1).contiguous()


y = torch.randn(2, 4, 4)
yp = y.permute(2, 0, 1).contiguous()

# print(y)
# print(yp)

x = torch.randn(2, 16, 512)
x = x.view(2, 16, 8, 64) # we can get exact same result if we use reshape, however we use .view() to save memory thanks to pytorch mechanism under the hood 

y = x.permute(0, 2, 1, 3) # we sometimes do this because the order from another software components (api contract) maybe different
out = y.contiguous() # ofc y eats some memory, but out will use y, but it's contiguous, so that when we send it to gpu/cpu, it's fast!!! 


print(out.shape)
print(out.is_contiguous())