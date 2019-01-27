import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

from prepareData import MyDataset
from net import Resnet50
import param as param

print(torch.__version__)


traindata = MyDataset()
data_loader = Data.DataLoader(
    dataset=traindata,      # torch TensorDataset format
    batch_size=param.batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)

net = Resnet50(param.num_class)
optimizer = optim.SGD(net.parameters(), lr=param.lr, momentum=0.9)  # optimize all cnn parameters
criterion = nn.CrossEntropyLoss()

for epoc in range(param.epoch):
    sumLoss = 0
    for step, (input, label) in enumerate(data_loader):
        optimizer.zero_grad()
        output = net.forward(input)
        print("output: ", output.size())
        print("label: ", label.size())
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()
        sumLoss += loss.item()
        if step % param.show_step == 0:
            print("epoc {}  step {}  sumloss:{%0.03f}".format(epoc, step, sumLoss))