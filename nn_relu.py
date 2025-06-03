import torch
from torch import nn
from torch.nn import Sigmoid

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms


input = torch.tensor([[1, -0.5],
                     [-1,3]])

output = torch.reshape(input,(-1,1,2,2))
print(output.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        output = self.sigmoid1(x)
        return output

tudui = Tudui()
output = tudui(input)
print(output)

dataset = datasets.CIFAR10("dataset", train=False, transform=transforms.ToTensor(), download=False)
writer = SummaryWriter('logs')
dataloader = DataLoader(dataset, batch_size=64)
step = 0
for data in dataloader:
    imgs,targets = data

    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
