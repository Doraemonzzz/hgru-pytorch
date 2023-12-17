import torch
from hgru import Luru

n = 512
b = 1
d = 32

dtype = torch.bfloat16
# dtype = torch.float16
# dtype = torch.float32

model = Luru(d).cuda().to(dtype)
print(model)

x = torch.randn(n, b, d).to(dtype).cuda().requires_grad_()
y1 = model(x)
y2 = model.forward_naive(x)


print(torch.norm(y1 - y2))

print("====================")

res = []

if x.grad != None:
    x.grad.data.zero_()
if model.input_proj.weight.grad != None:
    model.input_proj.weight.grad.data.zero_()
if model.gate[0].weight.grad != None:
    model.gate[0].weight.grad.data.zero_()
if model.gate[1].weight.grad != None:
    model.gate[1].weight.grad.data.zero_()
if model.out_proj.weight.grad != None:
    model.out_proj.weight.grad.data.zero_()
if model.norm.weight.grad != None:
    model.norm.weight.grad.data.zero_()

loss = (y1**2).sum()
loss.backward()

res.append(x.grad.data.clone())
res.append(model.input_proj.weight.grad.data.clone())
res.append(model.gate[0].weight.grad.data.clone())
res.append(model.gate[1].weight.grad.data.clone())
res.append(model.out_proj.weight.grad.data.clone())
res.append(model.norm.weight.grad.data.clone())

if x.grad != None:
    x.grad.data.zero_()
if model.input_proj.weight.grad != None:
    model.input_proj.weight.grad.data.zero_()
if model.gate[0].weight.grad != None:
    model.gate[0].weight.grad.data.zero_()
if model.gate[1].weight.grad != None:
    model.gate[1].weight.grad.data.zero_()
if model.out_proj.weight.grad != None:
    model.out_proj.weight.grad.data.zero_()
if model.norm.weight.grad != None:
    model.norm.weight.grad.data.zero_()

loss = (y2**2).sum()
loss.backward()

res.append(x.grad.data.clone())
res.append(model.input_proj.weight.grad.data.clone())
res.append(model.gate[0].weight.grad.data.clone())
res.append(model.gate[1].weight.grad.data.clone())
res.append(model.out_proj.weight.grad.data.clone())
res.append(model.norm.weight.grad.data.clone())


c = 6

for i in range(c):
    print(
        torch.norm(res[i] - res[i + c]),
    )
