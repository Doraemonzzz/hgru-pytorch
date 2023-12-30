import torch

from hgru import Hgru1d, Hgru2d, SHgruV36_Triton

n = 1024
b = 4
d = 1024
x_1d = torch.randn(b, n, d).cuda().to(torch.bfloat16)
hgru_1d = Hgru1d(embed_dim=d, causal=False).cuda().to(torch.bfloat16)
# x_2d = torch.randn(n, n, b, d).cuda()
# hgru_2d = Hgru2d(embed_dim=d, causal=False).cuda()

# print(x_1d.shape)

y_1d = hgru_1d(x_1d)
# y_2d = hgru_2d(x_2d)

print(y_1d.shape)



