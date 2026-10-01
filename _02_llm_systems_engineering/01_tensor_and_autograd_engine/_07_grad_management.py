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


x = torch.tensor([2.0], requires_grad=True)
y1 = x ** 2
y1.backward()
_print(x.grad.item())

y2 = x ** 3
y2.backward()
_print(x.grad.item())

x.grad.zero_()
_print(x.grad.item())

y2 = x ** 3
y2.backward()
_print(x.grad.item())


x_input = torch.tensor([5.0], requires_grad=True)


with torch.no_grad():
    out = x_input * 5.0

print("Output during no_grad context:")
print("out.requires_grad:", out.requires_grad)  # False!
print("out.grad_fn:", out.grad_fn)

w = torch.tensor([2.0], requires_grad=True)
a = w * 3.0

a_detached = a.detach()
print(a_detached.requires_grad)
print(a_detached.grad_fn)


# =====
w = torch.tensor(5.0, requires_grad=True)  

for i in range(200):
    if w.grad is not None:
        w.grad.zero_()
    
    loss = (w ** 2) * 2.0
    
    loss.backward()

    with torch.no_grad():
        w -= 0.01 * w.grad

    if i % 10 == 0:
        print(f'iteration-{i}: {loss.item()} | {w.grad.item()} | {w.item()}')