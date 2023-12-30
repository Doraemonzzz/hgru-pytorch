import torch
from hgru import SHgruV36_Triton

b = 1
n = 512
d = 64

dtype = torch.bfloat16
# dtype = torch.float16
dtype = torch.float32

model = SHgruV36_Triton(d, expand_ratio=16).cuda().to(dtype)
lower_bound = -torch.rand(d).cuda().to(dtype)

x = torch.randn(b, n, d).to(dtype).cuda().requires_grad_()
y1 = model(x, lower_bound)
y2 = model.forward_naive(x, lower_bound)


print(torch.norm(y1 - y2))

print("====================")

res = []

if x.grad != None:
    x.grad.data.zero_()
if model.in_proj.weight.grad != None:
    model.in_proj.weight.grad.data.zero_()
if model.out_proj.weight.grad != None:
    model.out_proj.weight.grad.data.zero_()
if model.norm.weight.grad != None:
    model.norm.weight.grad.data.zero_()

loss = (y1**2).sum()
loss.backward()

res.append(x.grad.data.clone())
res.append(model.in_proj.weight.grad.data.clone())
res.append(model.out_proj.weight.grad.data.clone())
res.append(model.norm.weight.grad.data.clone())

if x.grad != None:
    x.grad.data.zero_()
if model.in_proj.weight.grad != None:
    model.in_proj.weight.grad.data.zero_()
if model.out_proj.weight.grad != None:
    model.out_proj.weight.grad.data.zero_()
if model.norm.weight.grad != None:
    model.norm.weight.grad.data.zero_()

loss = (y2**2).sum()
loss.backward()

res.append(x.grad.data.clone())
res.append(model.in_proj.weight.grad.data.clone())
res.append(model.out_proj.weight.grad.data.clone())
res.append(model.norm.weight.grad.data.clone())


c = 4

for i in range(c):
    print(
        torch.norm(res[i] - res[i + c]),
        torch.norm(res[i]),
    )
