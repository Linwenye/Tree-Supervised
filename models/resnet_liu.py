'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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


def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
    )


class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


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
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.AvgPool2d(4, 4)

        self.attention1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=64 * block.expansion
            ),
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=256 * block.expansion
            ),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

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
        fea1 = self.attention1(out)
        fea1 = fea1 * out
        out1_feature = self.scala1(fea1).view(x.size(0), -1)

        out = self.layer2(out)
        fea2 = self.attention2(out)
        fea2 = fea2 * out
        out2_feature = self.scala2(fea2).view(x.size(0), -1)

        out = self.layer3(out)
        fea3 = self.attention3(out)
        fea3 = fea3 * out
        out3_feature = self.scala3(fea3).view(x.size(0), -1)

        out = self.layer4(out)
        out4_feature = self.scala4(out).view(x.size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]


class BiResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(BiResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.in_planes = 64  # redefine to fix the re-making layer shape error
        self.layer1_ = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_ = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

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
        """ temporary using 4 branch; supervise half"""
        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)
        out1 = self.layer2(out1)
        out2 = self.layer3_(out1)
        out2_feature = F.avg_pool2d(self.layer4_(out2), 4)
        out2_feature = out2_feature.view(out2_feature.size(0), -1)

        out1 = self.layer3(out1)
        out1_feature = F.avg_pool2d(self.layer4(out1), 4)
        out1_feature = out1_feature.view(out1_feature.size(0), -1)

        out3 = self.layer1_(out)
        out3 = self.layer2_(out3)
        out4 = self.layer3_(out3)
        out4_feature = F.avg_pool2d(self.layer4_(out4), 4)
        out4_feature = out4_feature.view(out4_feature.size(0), -1)

        out3 = self.layer3(out3)
        out3_feature = F.avg_pool2d(self.layer4(out3), 4)
        out3_feature = out3_feature.view(out3_feature.size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]


class BiResNet_multibranch(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(BiResNet_multibranch, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.in_planes = 64  # redefine to fix the re-making layer shape error
        self.layer1_ = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_ = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

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

        out1 = self.layer1(out)
        out1 = self.layer2(out1)
        out2 = self.layer3_(out1)
        out2_feature = F.avg_pool2d(self.layer4_(out2), 4)
        out2_feature = out2_feature.view(out2_feature.size(0), -1)

        out1 = self.layer3(out1)
        out1_feature = F.avg_pool2d(self.layer4(out1), 4)
        out1_feature = out1_feature.view(out1_feature.size(0), -1)

        out3 = self.layer1_(out)
        out3 = self.layer2_(out3)
        out4 = self.layer3_(out3)
        out4_feature = F.avg_pool2d(self.layer4_(out4), 4)
        out4_feature = out4_feature.view(out4_feature.size(0), -1)

        out3 = self.layer3(out3)
        out3_feature = F.avg_pool2d(self.layer4(out3), 4)
        out3_feature = out3_feature.view(out3_feature.size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]


class BiResNet_detach(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(BiResNet_detach, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.in_planes = 64  # redefine to fix the re-making layer shape error
        self.layer1_ = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_ = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

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

        """ temporary using 4 branch; supervise half"""
        out = F.relu(self.bn1(self.conv1(x)))

        def set_requires_grad(module, val):
            for p in module.parameters():
                p.requires_grad = val

        # det_layer3_ = self.layer3_.weight.detach()

        out1 = self.layer1(out)
        out1 = self.layer2(out1)
        set_requires_grad(self.layer3_, False)
        set_requires_grad(self.layer4_, False)
        out2 = self.layer3_(out1)
        out2_feature = F.avg_pool2d(self.layer4_(out2), 4)
        set_requires_grad(self.layer3_, True)
        set_requires_grad(self.layer4_, True)
        out2_feature = out2_feature.view(out2_feature.size(0), -1)

        out1 = self.layer3(out1)
        out1_feature = F.avg_pool2d(self.layer4(out1), 4)
        out1_feature = out1_feature.view(out1_feature.size(0), -1)

        out3 = self.layer1_(out)
        out3 = self.layer2_(out3)
        out4 = self.layer3_(out3)
        out4_feature = F.avg_pool2d(self.layer4_(out4), 4)
        out4_feature = out4_feature.view(out4_feature.size(0), -1)

        set_requires_grad(self.layer3, False)
        set_requires_grad(self.layer4, False)
        out3 = self.layer3(out3)
        out3_feature = F.avg_pool2d(self.layer4(out3), 4)
        set_requires_grad(self.layer3, True)
        set_requires_grad(self.layer4, True)
        out3_feature = out3_feature.view(out3_feature.size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]


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
        self.layer2 = nn.ModuleList([self._make_blocks(block, 16*block.expansion, 32, num_blocks[1], stride=2) for _ in range(2)])
        self.layer3 = nn.ModuleList([self._make_blocks(block, 32*block.expansion, 64, num_blocks[2], stride=2) for _ in range(4)])
        self.linears = nn.ModuleList([nn.Linear(64 * block.expansion, num_classes) for _ in range(4)])

    def _make_blocks(self, block, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(block(in_planes, out_planes, stride))
        for i in range(num_blocks - 1):
            layers.append(block(out_planes*block.expansion, out_planes, 1))
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

class TreeCifarResNetCombine(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, batchnorm=True):
        """layer 1 as root version"""
        super(TreeCifarResNetCombine, self).__init__()
        if batchnorm:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.Sequential()
        self.layer1 = nn.ModuleList([self._make_blocks(block, 16, 16, num_blocks[0], stride=1)])
        self.layer2 = nn.ModuleList([self._make_blocks(block, 16*block.expansion, 32, num_blocks[1], stride=2) for _ in range(2)])
        self.layer3 = nn.ModuleList([self._make_blocks(block, 32*block.expansion, 64, num_blocks[2], stride=2) for _ in range(4)])
        self.linears = nn.ModuleList([nn.Linear(64 * block.expansion, num_classes) for _ in range(4)])
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
    def _make_blocks(self, block, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(block(in_planes, out_planes, stride))
        for i in range(num_blocks - 1):
            layers.append(block(out_planes*block.expansion, out_planes, 1))
        return nn.Sequential(*layers)

    def init_modules(self):
        self.layer3 = self._make_blocks(self.block, 32*self.block.expansion, 64, self.num_blocks[2], stride=2)
        self.linears = nn.Linear(64 * self.block.expansion, self.num_classes)

    def forward(self, x, branch=0, freeze=True):
        context = torch.no_grad if freeze else torch.enable_grad
        with context():
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1[0](out)
            out1 = self.layer2[branch](out)
        # out1.requires_grad = False
        # input(out1.requires_grad)
        out = self.layer3(out1)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linears(out)
        return out

class TreeCifarResNet_l4(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, batchnorm=True):
        """ Cifar ResNet with 4 layer"""
        super(TreeCifarResNet_l4, self).__init__()
        if batchnorm:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.Sequential()
        self.layer1 = nn.ModuleList([self._make_blocks(block, 16, 16, num_blocks[0], stride=1)])
        self.layer2 = nn.ModuleList([self._make_blocks(block, 16*block.expansion, 32, num_blocks[1], stride=2) for _ in range(2)])
        self.layer3 = nn.ModuleList([self._make_blocks(block, 32*block.expansion, 64, num_blocks[2], stride=2) for _ in range(4)])
        self.linears = nn.ModuleList([nn.Linear(64 * block.expansion, num_classes) for _ in range(4)])

    def _make_blocks(self, block, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(block(in_planes, out_planes, stride))
        for i in range(num_blocks - 1):
            layers.append(block(out_planes*block.expansion, out_planes, 1))
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


# class TreeCifarResNet_v2(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10, image_channels=3, batchnorm=True):
#         """conv1 as root version"""
#         super(TreeCifarResNetv2, self).__init__()
#         self.in_planes = 16
#         if batchnorm:
#             self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(16)
#         else:
#             self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
#             self.bn1 = nn.Sequential()
#         self.layer1 = [self._make_blocks(block, 16, num_blocks[0], stride=1)]
#         self.layer2 = [self._make_blocks(block, 32, num_blocks[1], stride=2)]
#         self.layer3 = self._make_blocks(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64 * block.expansion, num_classes)
#
#     def _make_blocks(self, block, in_planes, out_planes, num_blocks, stride):
#         layers = []
#         layers.append(block(in_planes, out_planes, stride))
#         for i in range(num_blocks - 1):
#             layers.append(block(out_planes, out_planes, 1))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, 8)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
#     def _forward(self, x, res):


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

def TreeCifarResNet32Combine(num_classes):
    return TreeCifarResNetCombine(BasicBlock, [5, 5, 5], num_classes)

def TreeCifarResNet20_v1(num_classes):
    return TreeCifarResNet_v1(BasicBlock, [3, 3, 3], num_classes)


def TreeCifarResNet44_v1(num_classes):
    return TreeCifarResNet_v1(BasicBlock, [7, 7, 7], num_classes)


def TreeCifarResNet56_v1(num_classes):
    return TreeCifarResNet_v1(Bottleneck, [9, 9, 9], num_classes)


def TreeCifarResNet110_v1(num_classes):
    return TreeCifarResNet_v1(Bottleneck, [18, 18, 18], num_classes)


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def BiResNet18(num_classes):
    return BiResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def BiResNet_detach18(num_classes):
    return BiResNet_detach(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    # net = BiResNet18(100)
    # net = TreeCifarResNet32_v1(100)
    net = TreeCifarResNet110_v1(10)
    # print(net)
    y = net(torch.randn(1, 3, 32, 32))
    print(sum(p.numel() for p in net.parameters()))
    print(y[0].size())

test()
