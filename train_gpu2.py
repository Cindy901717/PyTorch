import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from model import *

#准备数据集
train_data = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=torchvision.transforms.ToTensor())

#定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))

#利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model = nn.Sequential(
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
        x = self.model(x)
        return x
tudui  = Tudui()
tudui.to(device)

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
total_test_step = 0
#训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-----第 {} 轮训练开始-----".format(i+1))

    #训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        imgs = tudui(imgs)
        loss = loss_fn(imgs, targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = imgs.to(device)
            targets = targets.to(device)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui.state_dict(), "tudui_{}.pth".format(i+1))
    #torch.save(tudui(),"tudui_{}.pth".format(i+1))
    print("模型已保存")

writer.close()

