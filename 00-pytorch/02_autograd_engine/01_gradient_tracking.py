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



w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([2.0])

print(w.is_leaf)
print(b.is_leaf)
print(x.is_leaf)

y = w * b + x
loss = y**2

print(y.item())
print(loss.item())

print("\n--- Computational Graph Function Pointers ---")
print("loss.grad_fn:", loss.grad_fn)  # <PowBackward0>
print("y.grad_fn:", y.grad_fn)        # <AddBackward0>
print("w.grad_fn:", w.grad_fn)        # None 

# d(loss)/dw = d(loss)/dy * dy/dw
# d(loss)/dy = 2 * y = 14.0
# dy/dw = x = 2.0
# d(loss)/dw = 14.0 * 2.0 = 28.0
loss.backward()

print("\n--- Accumulated Gradients ---")
print("w.grad:", w.grad.item())  # Should be 28.0
print("b.grad:", b.grad.item())  # Should be 14.0
print("x.grad:", x.grad)         # None (requires_grad was False)

print("\n" + "="*50 + "\n")


a = torch.tensor([4.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

c = a ** 3 + 2 * b
_print(c)
# dc/da = 3*a^2
# dc/db = 2
# dL/dc = 1 / (2 * sqrt(c))
# dL/da = 1/(2*sqrt(c))*3*a^2   
# dL/db = 1/(2*sqrt(c))*2
L = c.sqrt()
L.backward()
print(a.grad.item())
print(b.grad.item())

