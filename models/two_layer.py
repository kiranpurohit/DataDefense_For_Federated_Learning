import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


#Parameters of this network will be sent to Net_test as its parameters
#\theta

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.linear1 = nn.Linear(3072, 1024) #W1, b1 = Wx + b
        self.linear2 = nn.Linear(1024, 10) #W2, b2
      
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        #out = F.relu(out)
        out = self.linear2(out)
        return out

##################################################################################
##################################################################################

def lincalc(inp,wt,bias):
  output = inp.matmul(wt.t())
  fop = output + bias
  ret = fop
  return ret




class lin(nn.Module):
  def __init__(self, wt, bs):
        super(lin, self).__init__()
        self.weight = wt
        self.bias = bs
  def forward(self, input):
        #return F.linear(input, self.weight, self.bias)
        return lincalc(input, self.weight, self.bias)



class Net_test(nn.Module):
    def __init__(self,paramsel):
        super(Net_test, self).__init__()
        self.lin1 = lin(paramsel['linear1.weight'].clone(),paramsel['linear1.bias'].clone())
        self.lin2 = lin(paramsel['linear2.weight'].clone(),paramsel['linear2.bias'].clone())

    def forward(self, x):
        out = x.view(x.size(0), -1)
        #print("Inside net test")
        out1 = self.lin1(out) #W1x + b1
        #print(out1.requires_grad)
        out2 = self.lin2(out1) #W2x + b2
        #print(out2.requires_grad)
        return out2


