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
testdata = MyDataset("test")

test_loader = Data.DataLoader(
    dataset=testdata,      # torch TensorDataset format
    batch_size=param.batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)
print("testdata:", testdata.__len__())
net = Resnet50(param.num_class)
modelfile = "model/resnet50_16w_epoc_10.pkl"
net.load_state_dict(torch.load(modelfile))

criterion = nn.MSELoss()
if param.is_cuda:
    device = torch.device('cuda:2')
    net.cuda()
    criterion.cuda()

testLoss = 0
result = [0 for _ in range(param.num_class)]
for t, (x, y) in enumerate(test_loader):
    if param.is_cuda:
        x = x.cuda()
        y = y.cuda()
    predict = net.forward(x)
    err =criterion(predict, y)
    testLoss += err.item()
    res = predict.mul(y)
    res = res.cpu().data.numpy()
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            result[j] += 1 if res[i][j] > 0 else 0  # if classification is right, set value 1
print("******************")
print("test loss: %0.03f}" % (testLoss))
print("result:", result)
avg_rate = 0
for k in range(len(attributes)):
    right_rate = float(result[k] / testdata.__len__())
    print(attributes[k] + " right rate is %0.03f" % right_rate)
    avg_rate += right_rate
avg_rate = float(avg_rate / param.num_class)
print("avarage right rate is %0.03f" % avg_rate)


