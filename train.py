import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.optim as optim

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
if param.is_cuda:
    device = torch.device('cuda:2')
    net.cuda()
    criterion.cuda()
for epoc in range(1, param.epoch):
    sumLoss = 0
    for step, (input, label) in enumerate(data_loader):
        optimizer.zero_grad()
        if param.is_cuda:
            input = input.cuda()
            label = label.cuda()
        output = net.forward(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        sumLoss += loss.item()
        # if step % param.show_step == 0:
            # print("train epoc %d  step %d  sumloss:%0.03f" % (epoc, step, sumLoss))
    print("train epoc %d  sumloss:%0.03f" % (epoc, sumLoss/traindata.__len__()))
    if epoc % param.val_step == 0:
        valLoss = 0
        result = [0 for _ in range(40)]
        for t, (x, y) in enumerate(val_loader):
            if param.is_cuda:
                x = x.cuda()
                y = y.cuda()
            predict = net.forward(x)
            err =criterion(predict, y)
            res = predict.mul(y)
            res = res.cpu().data.numpy()
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    result[j] += 1 if res[i][j] > 0 else 0  # if classification is right, set value 1
        print("******************")
        print("result:", result)
        for k in range(len(attributes)):
            right_rate = float(result[k] / valdata.__len__())
            print(attributes[k] + " right rate is %0.03f" % right_rate)
            valLoss += err.item()
        print("validation epoc %d valLoss: %0.03f}" % (epoc, valLoss))
        torch.save(net.state_dict(), "model/resnet50_2w_epoc_%d.pkl" % epoc)

