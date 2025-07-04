import torch
import torchvision
from torch import nn


vgg16 = torchvision.models.vgg16(pretrained=False)
#保存方式1, 模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

#保存方式2 ，模型参数（官方推荐）
torch.save(vgg16, "vgg16_method2.pth")

# 陷阱1
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        return x
tudui = Tudui()
torch.save(tudui,"tudui_method1.pth")


