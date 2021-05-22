'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from models import *
from configs import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100|cifar10")
parser.add_argument('--ckpt', default="data/tree_resnet32-1.pth", type=str, help="cifar100|cifar10")
parser.add_argument('--model', default="tree_resnet32_combine", type=str, help="resnet20|resnet32|mobilev3|wide")
# parser.add_argument('--weight_decay', default=1e-4, type=float, help='5e-4| 1e-4')
parser.add_argument('--gpus', default=1, type=int)
parser.add_argument('--tactic', default=-1, type=int)
parser.add_argument('--epoch', default=150, type=int, help="training epochs")
parser.add_argument('--down_epoch', type=int, nargs='+', default=[50, 80],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--freeze_before', default=200, type=int)
args = parser.parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
using_wanbd = True
start_time = time.time()
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
# if args.model == 'mobilev3':
#     net = MobileNetV3_Large(num_class)
#     config = config_mobilev3
# elif args.model == 'mobilev2':
#     net = MobileNetV2(num_class)
#     config = config_mobilev3
# elif args.model == 'wide':
#     net = Wide_ResNet(28,10,0,num_class)
#     config = config_wide_resnet
# elif args.model == 'resnet44':
#     net = CifarResNet44(num_class)
#     config = config_resnet
# elif args.model == 'resnet110':
#     net = CifarResNet110(num_class)
#     config = config_resnet
# elif args.model == 'resnet20':
#     net = CifarResNet20(num_class)
#     config = config_resnet
# elif args.model == 'tree_resnet32_combine':
#     net = TreeCifarResNet32Combine(num_class)
#     config = config_resnet

# else:
#     raise NameError

config = config_tree_resnet
net1 = CifarResNet32(num_class)
net2 = TreeCifarResNet32_v1(num_class)
net3 = TreeCifarResNet32_v1(num_class)

if device == 'cuda':
    print('Using cuda')
    net1 = torch.nn.DataParallel(net1)
    net2 = torch.nn.DataParallel(net2)
    net3 = torch.nn.DataParallel(net3)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(0)
    # torch.manual_seed(time.time())


# if args.ckpt is not '':
    # net.load_state_dict(torch.load(args.ckpt))

net1.load_state_dict(torch.load('./checkpoints/resnet32replace.pth'))
net2.load_state_dict(torch.load('./checkpoints/tree_resnet32-replace.pth'))
# net3.load_state_dict(torch.load('./checkpoints/tree_resnet32.pth'))
net3.load_state_dict(torch.load('./checkpoints/tree_resnet32.pth'))

# net.module.init_modules()
net1 = net1.to(device)
net2 = net2.to(device)
net3 = net3.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net1.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=config.weight_decay)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config.batch_size*args.gpus, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=config.batch_size*args.gpus, shuffle=False, num_workers=8)

if using_wanbd:
    import wandb
    wandb.init(project="combine")
    wandb.watch(net1,log="all")


def adjust_lr(epoch):
    if epoch in args.down_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

# Training
def train(epoch, net):
    GREEN = '\033[92m'
    RED = '\033[93m'
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # freeze = True if epoch < args.freeze_before else False
    freeze = False
    color = GREEN if freeze else RED
    e = '\nEpoch: %d  ' % epoch
    print(e + color + ('Freeze' if freeze else 'Not Freeze') + '\033[0m')
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # outputs = net(inputs, freeze=freeze)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('train acc:',correct/total*100)
    if using_wanbd:
        wandb.log({'train acc':correct/total*100})
    torch.save(net.state_dict(), 'checkpoints/exchange.last.pth')

def test(epoch, net):
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
    if using_wanbd:
        wandb.log({'test acc':correct/total*100})
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('best..')
        best_acc = acc
    #     torch.save(net.state_dict(), "./checkpoints/" + str(args.model) + "-{}-best.pth".format(args.tactic))


# for epoch in range(start_epoch, start_epoch+args.epoch):
#     start_t = time.time()
#     adjust_lr(epoch)
#     train(epoch)
#     if epoch<5:
#         print('train time:',time.time()-start_t)
#     test(epoch)
#     if epoch<5:
#         print('train and test time',time.time()-start_t)
# print('Finished, best acc',best_acc)


if __name__ == '__main__':

    # net1.module.bn1 = net1.module.bn1
    # net1.module.conv1 = net1.module.conv1
    # net1.module.layer1 = net1.module.layer1[0]
    # net1.module.bn1 = net1.module.bn1
    # net1.module.conv1 = net1.module.conv1
    # net1.module.layer1 = net1.module.layer1[0]
    net1.module.init_modules()
    net1.cuda()
    # net1.module.layer2 = net2.module.layer2[0]
    # net1.module.layer3 = net2.module.layer3[0]
    # net1.module.linear = net2.module.linears[0]
    test(0, net1)
    for epoch in range(args.epoch):
        train(epoch, net1)
        test(0, net1)
        # input()
        adjust_lr(epoch+2)
    print('Finished, best acc',best_acc)
    print('Time {:.2f} h'.format((time.time() - start_time) / 3600.))
    # test_tree(net1)   