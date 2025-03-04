# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init_mff(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init_mff(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self, layer, snapshot):
        super(ResNet, self).__init__()
        self.snapshot = snapshot
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, layer[0], stride=1, dilation=1)
        self.layer2 = self.make_layer(128, layer[1], stride=2, dilation=1)
        self.layer3 = self.make_layer(256, layer[2], stride=2, dilation=1)
        self.layer4 = self.make_layer(512, layer[3], stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load(self.snapshot), strict=False)


def resnet50():
    return ResNet([3, 4, 6, 3], '/data/zxk/code/code03/weights/resnet50-19c8e357.pth')


class MFFModel(nn.Module):
    def __init__(self, cfg):
        super(MFFModel, self).__init__()
        self.cfg = cfg
        self.bkbone = resnet50()
        self.fc51 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.bn51 = nn.BatchNorm2d(128)
        self.fc52 = nn.Conv2d(128, 2048, kernel_size=1, stride=1, padding=0)
        self.fc41 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.bn41 = nn.BatchNorm2d(128)
        self.fc42 = nn.Conv2d(128, 1024, kernel_size=1, stride=1, padding=0)
        self.fc31 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn31 = nn.BatchNorm2d(128)
        self.fc32 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.linear5 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(128)
        self.linear4 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.linear3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.linear = nn.Conv2d(128, self.cfg.TRAIN.NUM_CLASS, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def forward(self, x, mode="train"):
        out2, out3, out4, out5 = self.bkbone(x)

        out5a = F.relu(self.bn51(self.fc51(out5.mean(dim=(2, 3), keepdim=True))), inplace=True)
        out4a = F.relu(self.bn41(self.fc41(out4.mean(dim=(2, 3), keepdim=True))), inplace=True)
        out3a = F.relu(self.bn31(self.fc31(out3.mean(dim=(2, 3), keepdim=True))), inplace=True)
        vector = out5a * out4a * out3a
        out5 = torch.sigmoid(self.fc52(vector)) * out5
        out4 = torch.sigmoid(self.fc42(vector)) * out4
        out3 = torch.sigmoid(self.fc32(vector)) * out3

        out5 = F.relu(self.bn5(self.linear5(out5)), inplace=True)
        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.relu(self.bn4(self.linear4(out4)), inplace=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out3 = F.relu(self.bn3(self.linear3(out3)), inplace=True)

        if mode == 'train':
            pred1 = F.dropout(out5, p=0.5)
            pred1 = F.relu(self.linear(pred1), inplace=True)

            pred2 = F.dropout(out5 * out4 * out3, p=0.5)
            pred2 = F.relu(self.linear(pred2), inplace=True)
            return pred1, pred2
        else:
            pred1 = F.relu(self.linear(out5), inplace=True)
            pred2 = out5 * out4 * out3
            pred2 = F.relu(self.linear(pred2), inplace=True)
            return pred1, pred2

    def initialize(self):
        weight_init_mff(self)
