import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.optim as optim
import matplotlib.pyplot as plt

from prepareData import MyDataset
from net import Resnet50
import param as param

attributes = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald Bangs",
              "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
              "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
              "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
              "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
              "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
valdata = MyDataset("validation")
traindata = MyDataset("train")
data_loader = Data.DataLoader(
    dataset=traindata,      # torch TensorDataset format
    batch_size=param.batch_size,      # mini batch size"
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)
val_loader = Data.DataLoader(
    dataset=valdata,      # torch TensorDataset format
    batch_size=param.batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)
print("traindata:", traindata.__len__())
print("validation:", valdata.__len__())
net = Resnet50(param.num_class)
optimizer = optim.SGD(net.parameters(), lr=param.lr, momentum=0.9)  # optimize all cnn parameters
criterion = nn.MSELoss()

device = torch.device("cuda:0,1" if param.is_cuda else "cpu")
if param.multi_gpu:
    net = torch.nn.DataParallel(net)
criterion.to(device)
net.to(device)

trainLoss = []
validLoss = []
validX = []
for epoc in range(1, param.epoch):
    sumLoss = 0
    for step, (input, label) in enumerate(data_loader):
        optimizer.zero_grad()
        input = input.to(device)
        label = label.to(device)
        output = net.forward(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if param.multi_gpu:
            sumLoss += loss.mean().item()
        else:
            sumLoss += loss.item()
    trainLoss.append(sumLoss/traindata.__len__())
    print("train epoc %d  sumloss:%0.05f" % (epoc, sumLoss/traindata.__len__()))
    if epoc % param.val_step == 0:
        valLoss = 0
        result = [0 for _ in range(40)]
        for t, (x, y) in enumerate(val_loader):
            if param.is_cuda:
                x = x.cuda()
                y = y.cuda()
            predict = net.forward(x)
            err =criterion(predict, y)
            if param.multi_gpu:
                valLoss += err.mean().item()
            else:
                valLoss += err.item()
        validLoss.append(valLoss/valdata.__len__())
        validX.append(epoc)
    if epoc % param.save_model == 0:
        if param.multi_gpu:
            torch.save(net.module.state_dict(), "model/resnet50_16w_epoc_%d.pkl" % epoc)
        else:
            torch.save(net.state_dict(), "model/resnet50_16w_epoc_%d.pkl" % epoc)
# show loss
plt.cla()
plt.plot(trainLoss)
plt.xlabel("epoc")
plt.ylabel("train_loss")
plt.savefig("dataset/cache/train_loss.jpg")
plt.cla()
plt.plot(validX, validLoss)
plt.xlabel("epoc")
plt.ylabel("valid_loss")
plt.savefig("dataset/cache/valid_loss.jpg")
