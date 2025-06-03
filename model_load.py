import torch
import torchvision
from torch import nn
from model_save import *

#方式1 -》保存方式1，加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

#方式2，加载模型
# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

#陷阱
# class Tudui(nn.Module):
#      def __init__(self):
#          super(Tudui,self).__init__()
#          self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)
#
#      def forward(self,x):
#          x = self.conv1(x)
#          return x

# tudui = Tudui()
model = torch.load("tudui_method1.pth")
print(model)


