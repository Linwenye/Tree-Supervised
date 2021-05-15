'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

import os
import argparse

from models import *
from configs import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100|cifar10")
parser.add_argument('--model', default="resnet20", type=str, help="resnet20|resnet32|mobilev3|wide")
# parser.add_argument('--weight_decay', default=1e-4, type=float, help='5e-4| 1e-4')
parser.add_argument('--gpus', default=4, type=int)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
scaler = GradScaler()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='./data',train=True,download=True,transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data',train=False,download=True,transform=transform_test)
    num_class = 100
elif args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform_test)
    num_class = 10

# Model
print('==> Building model..')
if args.model == 'mobilev3':
    net = MobileNetV3_Large(num_class)
    config = config_mobilev3
elif args.model == 'mobilev2':
    net = MobileNetV2(num_class)
    config = config_mobilev3
elif args.model == 'wide':
    net = Wide_ResNet(28,10,0,num_class)
    config = config_wide_resnet
elif args.model == 'resnet44':
    net = CifarResNet44(num_class)
    config = config_resnet
elif args.model == 'resnet110':
    net = CifarResNet110(num_class)
    config = config_resnet
elif args.model == 'resnet20':
    net = CifarResNet20(num_class)
    config = config_resnet
else:
    raise NameError
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=config.weight_decay)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config.batch_size*args.gpus, shuffle=True, num_workers=5)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=config.batch_size*args.gpus, shuffle=False, num_workers=5)

def adjust_lr(epoch):
    if epoch in config.down_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

# Training
def train(epoch):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('train acc:',correct/total*100)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('test_loss:',test_loss)
    print('test Acc:', correct/total*100)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('best..')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+config.epoch):
    start_t = time.time()
    adjust_lr(epoch)
    train(epoch)
    if epoch<5:
        print('train time:',time.time()-start_t)
    test(epoch)
    if epoch<5:
        print('train and test time',time.time()-start_t)
print('Finished, best acc',best_acc)