import torch
from hgru import Hgru1d

n = 512
b = 1
d = 32

dtype = torch.bfloat16
# dtype = torch.float16
# dtype = torch.float32

model = Hgru1d(d, use_triton=False).cuda().to(dtype)

x = torch.randn(n, b, d).to(dtype).cuda().requires_grad_()
y1 = model(x)
y2 = model.forward_naive(x)
y3 = model.forward_triton(x)


print(torch.norm(y1 - y2), torch.norm(y1 - y3))

print("====================")

res = []

if x.grad != None:
    x.grad.data.zero_()
if model.input_proj.weight.grad != None:
    model.input_proj.weight.grad.data.zero_()
if model.lambda_proj.weight.grad != None:
    model.lambda_proj.weight.grad.data.zero_()
if model.theta.grad != None:
    model.theta.grad.data.zero_()
if model.gate.weight.grad != None:
    model.gate.weight.grad.data.zero_()
if model.out_proj.weight.grad != None:
    model.out_proj.weight.grad.data.zero_()
if model.norm.weight.grad != None:
    model.norm.weight.grad.data.zero_()

loss = (y1**2).sum()
loss.backward()

res.append(x.grad.data.clone())
res.append(model.input_proj.weight.grad.data.clone())
res.append(model.lambda_proj.weight.grad.data.clone())
res.append(model.theta.grad.data.clone())
res.append(model.gate.weight.grad.data.clone())
res.append(model.out_proj.weight.grad.data.clone())
res.append(model.norm.weight.grad.data.clone())

if x.grad != None:
    x.grad.data.zero_()
if model.input_proj.weight.grad != None:
    model.input_proj.weight.grad.data.zero_()
if model.lambda_proj.weight.grad != None:
    model.lambda_proj.weight.grad.data.zero_()
if model.theta.grad != None:
    model.theta.grad.data.zero_()
if model.gate.weight.grad != None:
    model.gate.weight.grad.data.zero_()
if model.out_proj.weight.grad != None:
    model.out_proj.weight.grad.data.zero_()
if model.norm.weight.grad != None:
    model.norm.weight.grad.data.zero_()

loss = (y2**2).sum()
loss.backward()

res.append(x.grad.data.clone())
res.append(model.input_proj.weight.grad.data.clone())
res.append(model.lambda_proj.weight.grad.data.clone())
res.append(model.theta.grad.data.clone())
res.append(model.gate.weight.grad.data.clone())
res.append(model.out_proj.weight.grad.data.clone())
res.append(model.norm.weight.grad.data.clone())


if x.grad != None:
    x.grad.data.zero_()
if model.input_proj.weight.grad != None:
    model.input_proj.weight.grad.data.zero_()
if model.lambda_proj.weight.grad != None:
    model.lambda_proj.weight.grad.data.zero_()
if model.theta.grad != None:
    model.theta.grad.data.zero_()
if model.gate.weight.grad != None:
    model.gate.weight.grad.data.zero_()
if model.out_proj.weight.grad != None:
    model.out_proj.weight.grad.data.zero_()
if model.norm.weight.grad != None:
    model.norm.weight.grad.data.zero_()

loss = (y3**2).sum()
loss.backward()

res.append(x.grad.data.clone())
res.append(model.input_proj.weight.grad.data.clone())
res.append(model.lambda_proj.weight.grad.data.clone())
res.append(model.theta.grad.data.clone())
res.append(model.gate.weight.grad.data.clone())
res.append(model.out_proj.weight.grad.data.clone())
res.append(model.norm.weight.grad.data.clone())

c = 7

for i in range(c):
    print(
        torch.norm(res[i] - res[i + c]),
        torch.norm(res[i] - res[i + 2 * c]),
        torch.norm(res[i]),
    )
