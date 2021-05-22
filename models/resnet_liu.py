'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TreeCifarResNet_v1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, batchnorm=True):
        """layer 1 as root version"""
        super(TreeCifarResNet_v1, self).__init__()
        if batchnorm:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.Sequential()
        self.layer1 = nn.ModuleList([self._make_blocks(block, 16, 16, num_blocks[0], stride=1)])
        self.layer2 = nn.ModuleList(
            [self._make_blocks(block, 16 * block.expansion, 32, num_blocks[1], stride=2) for _ in range(2)])
        self.layer3 = nn.ModuleList(
            [self._make_blocks(block, 32 * block.expansion, 64, num_blocks[2], stride=2) for _ in range(4)])
        self.linears = nn.ModuleList([nn.Linear(64 * block.expansion, num_classes) for _ in range(4)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_blocks(self, block, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(block(in_planes, out_planes, stride))
        for i in range(num_blocks - 1):
            layers.append(block(out_planes * block.expansion, out_planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out)
        out1 = self.layer2[0](out)
        out3 = self.layer2[1](out)

        out2 = self.layer3[1](out1)
        out1 = self.layer3[0](out1)
        out4 = self.layer3[3](out3)
        out3 = self.layer3[2](out3)
        # out = self.layer3(out)
        res = [out1, out2, out3, out4]
        for i in range(len(res)):
            res[i] = F.avg_pool2d(res[i], 8)
            res[i] = res[i].view(res[i].size(0), -1)
            res[i] = self.linears[i](res[i])
        return res


class TreeResNet18_(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, batchnorm=True):
        """layer 1 as root version"""
        super(TreeResNet18_, self).__init__()
        if batchnorm:
            self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.Sequential()
        self.layer1 = nn.ModuleList(
            [nn.Sequential(block(64, 64, 1))])
        self.layer2 = nn.ModuleList(
            [nn.Sequential(block(64, 64, 1),
                           block(64, 128, 2),
                           block(128, 128, 1)) for _ in range(2)])
        self.layer3 = nn.ModuleList(
            [nn.Sequential(block(128, 256, 2),
                           block(256, 256, 1),
                           block(256, 512, 2),
                           block(512, 512, 1)) for _ in range(4)])

        self.linears = nn.ModuleList([nn.Linear(512, num_classes) for _ in range(4)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_blocks(self, block, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(block(in_planes, out_planes, stride))
        for i in range(num_blocks - 1):
            layers.append(block(out_planes * block.expansion, out_planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out)
        out1 = self.layer2[0](out)
        out3 = self.layer2[1](out)

        out2 = self.layer3[1](out1)
        out1 = self.layer3[0](out1)
        out4 = self.layer3[3](out3)
        out3 = self.layer3[2](out3)
        # out = self.layer3(out)
        res = [out1, out2, out3, out4]
        for i in range(len(res)):
            res[i] = F.avg_pool2d(res[i], 4)
            res[i] = res[i].view(res[i].size(0), -1)
            res[i] = self.linears[i](res[i])
        return res


class TreeResNet50_(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, batchnorm=True):
        """layer 1 as root version"""
        super(TreeResNet50_, self).__init__()
        if batchnorm:
            self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.Sequential()
        self.layer1 = nn.ModuleList(
            [nn.Sequential(block(64, 64, 1), block(64*block.expansion, 64, 1), block(64*block.expansion, 64, 1))])
        self.layer2 = nn.ModuleList(
            [nn.Sequential(block(64*block.expansion, 128, 2), block(128*block.expansion, 128, 1), block(128*block.expansion, 128, 1),
                           block(128*block.expansion, 128, 1), block(128*block.expansion, 256, 2),
                           block(256*block.expansion, 256, 1)) for _ in range(2)])
        self.layer3 = nn.ModuleList(
            [nn.Sequential(
                block(256*block.expansion, 256, 1), block(256*block.expansion, 256, 1), block(256*block.expansion, 256, 1), block(256*block.expansion, 256, 1),
                block(256*block.expansion, 512, 2),
                block(512*block.expansion, 512, 1), block(512*block.expansion, 512, 1)) for _ in range(4)])

        self.linears = nn.ModuleList([nn.Linear(512*block.expansion, num_classes) for _ in range(4)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_blocks(self, block, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(block(in_planes, out_planes, stride))
        for i in range(num_blocks - 1):
            layers.append(block(out_planes * block.expansion, out_planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out)
        out1 = self.layer2[0](out)
        out3 = self.layer2[1](out)

        out2 = self.layer3[1](out1)
        out1 = self.layer3[0](out1)
        out4 = self.layer3[3](out3)
        out3 = self.layer3[2](out3)
        # out = self.layer3(out)
        res = [out1, out2, out3, out4]
        for i in range(len(res)):
            res[i] = F.avg_pool2d(res[i], 4)
            res[i] = res[i].view(res[i].size(0), -1)
            res[i] = self.linears[i](res[i])
        return res


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, batchnorm=True):
        super(CifarResNet, self).__init__()
        self.in_planes = 16
        if batchnorm:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.Sequential()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def CifarResNet20(num_classes):
    return CifarResNet(BasicBlock, [3, 3, 3], num_classes)


def CifarResNet32(num_classes):
    return CifarResNet(BasicBlock, [5, 5, 5], num_classes)


def CifarResNet44(num_classes):
    return CifarResNet(BasicBlock, [7, 7, 7], num_classes)


def CifarResNet56(num_classes):
    return CifarResNet(BasicBlock, [9, 9, 9], num_classes)


def CifarResNet110(num_classes):
    return CifarResNet(BasicBlock, [18, 18, 18], num_classes)


def TreeCifarResNet32_v1(num_classes):
    return TreeCifarResNet_v1(BasicBlock, [5, 5, 5], num_classes)


def TreeCifarResNet20_v1(num_classes):
    return TreeCifarResNet_v1(BasicBlock, [3, 3, 3], num_classes)


def TreeCifarResNet44_v1(num_classes):
    return TreeCifarResNet_v1(BasicBlock, [7, 7, 7], num_classes)


def TreeCifarResNet56_v1(num_classes):
    return TreeCifarResNet_v1(Bottleneck, [9, 9, 9], num_classes)


def TreeCifarResNet110_v1(num_classes):
    return TreeCifarResNet_v1(Bottleneck, [18, 18, 18], num_classes)


def TreeResNet18(num_classes):
    return TreeResNet18_(BasicBlock, [2, 2, 2, 2], num_classes)


def TreeResNet50(num_classes):
    return TreeResNet50_(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = TreeResNet50(100)
    y = net(torch.randn(1, 3, 32, 32))
    print(sum(p.numel() for p in net.parameters()))
    print(y[0].size())


test()
