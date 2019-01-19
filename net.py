import torch.nn as nn
import torch.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Bottleneck, self.__init__())
        self.conv1 = nn.Conv2d(in_planes, in_planes, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = nn.ReLU(out1, True)

        out2 = self.conv2(out1)
        out2 = self.bn1(out2)
        out2 = nn.ReLU(out2, True)

        out3 = self.conv3(out2)
        out3 = self.bn2(out3)

        out3 = out3 + x
        out3 = nn.ReLU(out3, True)
        return out3

class Resnet50(nn.Module):
    def __init__(self, classes):
        super(Resnet50, self.__init__())
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(Bottleneck, 3, 64, stride=2)
        self.layer2 = self._make_layer(Bottleneck, 3, 128, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 3, 256, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 3, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, classes)

    def _make_layer(self, bottleneck, num, in_planes, stride):
        layers = []
        for _ in range(num):
            layers.append(bottleneck(in_planes, in_planes*4))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = nn.ReLU(out1, True)
        out1 = self.pool1(out1)

        out1 = self.layer1(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = self.avgpool(out4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out










