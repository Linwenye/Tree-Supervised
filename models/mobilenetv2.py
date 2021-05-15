'''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TreeMobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        [(1, 16, 1, 1),
         (6, 24, 2, 1)],  # NOTE: change stride 2 -> 1 for CIFAR10
        [(6, 32, 3, 2),
         (6, 64, 4, 2),
         (6, 96, 3, 1)],
        [(6, 160, 3, 2),
         (6, 320, 1, 1)]]

    def __init__(self, num_classes=10):
        super(TreeMobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = nn.ModuleList([self._make_layers(32, self.cfg[0])])
        self.layer2 = nn.ModuleList([self._make_layers(24, self.cfg[1]) for _ in range(2)])
        self.layer3 = nn.ModuleList([self._make_layers(96, self.cfg[2]) for _ in range(4)])
        self.conv2s = nn.ModuleList(
            [nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(4)])
        self.bn2s = nn.ModuleList([nn.BatchNorm2d(1280) for _ in range(4)])
        self.linears = nn.ModuleList([nn.Linear(1280, num_classes) for _ in range(4)])

    def _make_layers(self, in_planes, cfg):
        layers = []
        for expansion, out_planes, num_blocks, stride in cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
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

        res = [out1, out2, out3, out4]
        for i in range(4):
            res[i] = F.relu(self.bn2s[i](self.conv2s[i](res[i])))
            # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
            res[i] = F.avg_pool2d(res[i], 4)
            res[i] = res[i].view(res[i].size(0), -1)
            res[i] = self.linears[i](res[i])
        return res



class TreeMobileNetV2_image(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        [(1, 16, 1, 1),
         (6, 24, 2, 2)],  # NOTE: change stride 2 -> 1 for CIFAR10
        [(6, 32, 3, 2),
         (6, 64, 4, 2),
         (6, 96, 3, 1)],
        [(6, 160, 3, 2),
         (6, 320, 1, 1)]]

    def __init__(self, num_classes=1000):
        super(TreeMobileNetV2_image, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = nn.ModuleList([self._make_layers(32, self.cfg[0])])
        self.layer2 = nn.ModuleList([self._make_layers(24, self.cfg[1]) for _ in range(2)])
        self.layer3 = nn.ModuleList([self._make_layers(96, self.cfg[2]) for _ in range(4)])
        self.conv2s = nn.ModuleList(
            [nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(4)])
        self.bn2s = nn.ModuleList([nn.BatchNorm2d(1280) for _ in range(4)])
        self.linears = nn.ModuleList([nn.Linear(1280, num_classes) for _ in range(4)])

    def _make_layers(self, in_planes, cfg):
        layers = []
        for expansion, out_planes, num_blocks, stride in cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
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

        res = [out1, out2, out3, out4]
        for i in range(4):
            res[i] = F.relu(self.bn2s[i](self.conv2s[i](res[i])))
            # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
            res[i] = F.avg_pool2d(res[i], 7)
            res[i] = res[i].view(res[i].size(0), -1)
            res[i] = self.linears[i](res[i])
        return res

def test():
    net = TreeMobileNetV2_image()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(sum(p.numel() for p in net.parameters()))
    print(net)
    print(y[0].size())

# test()
