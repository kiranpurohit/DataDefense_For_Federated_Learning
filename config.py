import torch
import torch.nn as nn
torch.manual_seed(0)

theta = torch.empty((1,4), requires_grad=True).cuda()
nn.init.xavier_normal_(theta)
theta.retain_grad()

w1 = torch.empty((256,512), requires_grad=True).cuda()
nn.init.xavier_normal_(w1)
w1.retain_grad()

w2 = torch.empty((10, 256), requires_grad=True).cuda()
nn.init.xavier_normal_(w2)
w2.retain_grad()

w3 = torch.empty((256,20), requires_grad=True).cuda()
nn.init.xavier_normal_(w3)
w3.retain_grad()

w4 = torch.empty((1,256), requires_grad=True).cuda()
nn.init.xavier_normal_(w4)
w4.retain_grad()
'''
theta = torch.randn(4, requires_grad=True).cuda()
theta.retain_grad()

w1 = torch.randn((256,512), requires_grad=True).cuda()
w1.retain_grad()

w2 = torch.randn((10, 256), requires_grad=True).cuda()
w2.retain_grad()

w3 = torch.randn((256,20), requires_grad=True).cuda()
w3.retain_grad()

w4 = torch.randn(256, requires_grad=True).cuda()
w4.retain_grad()
'''
