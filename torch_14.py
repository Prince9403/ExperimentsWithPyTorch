import torch
import torch.nn as nn


a = torch.tensor(10.0, requires_grad=True)

optimizer = torch.optim.Adam([a], lr=0.01)

print("a:", a)
print("torch.abs(a):", torch.abs(a))
print("torch.abs(a-7.0):", torch.abs(a-7.0))

for i in range(305):
    b = a - 7.0
    loss = nn.functional.relu(b)
    if i >= 299:
        print("i=", i, "a=", a, "b=", b, "loss=", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


