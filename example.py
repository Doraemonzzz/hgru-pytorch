import torch

from hgru import Hgru1d, Hgru2d

n = 10
b = 2
d = 32
x_1d = torch.randn(n, b, d).cuda().to(torch.bfloat16)
hgru_1d = Hgru1d(embed_dim=d, causal=False).cuda().to(torch.bfloat16)
x_2d = torch.randn(n, n, b, d).cuda()
hgru_2d = Hgru2d(embed_dim=d, causal=False).cuda()

print(x_1d.shape)

y_1d = hgru_1d(x_1d)
y_2d = hgru_2d(x_2d)

print(y_1d.shape, y_2d.shape)
