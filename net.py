import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 1, 1, False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_planes, out_planes * self.expansion, 1, 1, False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = F.relu(out1, True)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = F.relu(out2, True)

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
       
        if self.downsample is not None:
            identity = self.downsample(x)
        # print("   neck out:", out3.size())
        # print("   neck identity:", identity.size())
        out3 = out3 + identity
        out3 = F.relu(out3, True)
        return out3

class Resnet50(nn.Module):
    def __init__(self, classes):
        super(Resnet50, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(Bottleneck, 3, 64, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 4, 128, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 6, 256, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 3, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, classes)

    def _make_layer(self, bottleneck, num, planes, stride):
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes * bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*bottleneck.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * bottleneck.expansion)
            )
        layers.append(bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * bottleneck.expansion
        for _ in range(num):
            layers.append(bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = F.relu(out1, inplace=True)
        out1 = self.pool1(out1)

        out1 = self.layer1(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = self.avgpool(out4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        # print("x:", x.size())
        # print("out0: ", out1.size())
        # print("layer1:", out1.size())
        # print("layer2: ", out2.size())
        # print("layer3: ", out3.size())
        # print("layer4: ", out4.size())
        # print("fc: ", out.size())
        return out










