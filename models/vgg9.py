import torch
import torch.nn as nn
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), 
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(64, 128, kernel_size=3, padding=1), 
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.AvgPool2d(kernel_size=1, stride=1))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

#####################################################################################
class conv(nn.Module):
  def __init__(self, wt, bs):
        super(conv, self).__init__()
        self.weight = wt
        self.bias = bs
  def forward(self, input):
        out = F.conv2d(input, self.weight, self.bias, stride=1, padding=1, dilation=1, groups=1)
        return out
######################################################################################
class lin(nn.Module):
  def __init__(self, wt, bs):
        super(lin, self).__init__()
        self.weight = wt
        self.bias = bs
  def forward(self, input):
        out = F.linear(input, self.weight, self.bias)
        return out
######################################################################################

class Net_test(nn.Module):
    def __init__(self,paramsel):
        super(Net_test, self).__init__()
        self.fea0 = conv(paramsel['features.0.weight'], paramsel['features.0.bias'])
        self.fea3 = conv(paramsel['features.3.weight'], paramsel['features.3.bias'])
        self.fea6 = conv(paramsel['features.6.weight'], paramsel['features.6.bias'])
        self.fea8 = conv(paramsel['features.8.weight'], paramsel['features.8.bias'])
        self.fea11 = conv(paramsel['features.11.weight'], paramsel['features.11.bias'])
        self.fea13 = conv(paramsel['features.13.weight'], paramsel['features.13.bias'])
        self.fea16 = conv(paramsel['features.16.weight'], paramsel['features.16.bias'])
        self.fea18 = conv(paramsel['features.18.weight'], paramsel['features.18.bias'])
        self.clasi = lin(paramsel['classifier.weight'], paramsel['classifier.bias'])

    def forward(self, x):
        #print("Inside net test")
        out = F.relu(self.fea0(x), inplace=True)  
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.fea3(out), inplace=True)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.fea6(out), inplace=True) 
        out = F.relu(self.fea8(out), inplace=True) 
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.fea11(out), inplace=True) 
        out = F.relu(self.fea13(out), inplace=True) 
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.fea16(out), inplace=True) 
        out = F.relu(self.fea18(out), inplace=True) 
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.avg_pool2d(out, kernel_size=1, stride=1)
        out = out.view(out.size(0), -1)
        out = self.clasi(out) 
        #print(out.requires_grad)
        return out


