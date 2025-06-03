from torch import nn
from torch.nn import Sequential, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

dataset = datasets.CIFAR10("dataset", train=False, transform=transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=1)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model1 = Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )



    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs,targets)
    result_loss.backward()
    print("ok")


