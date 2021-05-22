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
parser.add_argument('--down_epoch', type=int, nargs='+', default=[60, 90, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--freeze_before', default=200, type=int)
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--temperature', default=3.0, type=float)

args = parser.parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_single = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
using_wanbd = False

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

if device == 'cuda':
    print('Using cuda')
    net1 = torch.nn.DataParallel(net1)
    net2 = torch.nn.DataParallel(net2)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(0)


# if args.ckpt is not '':
    # net.load_state_dict(torch.load(args.ckpt))

net1.load_state_dict(torch.load('./checkpoints/resnet32replace.pth'))
net2.load_state_dict(torch.load('./checkpoints/tree_resnet32-replace.pth'))

# net.module.init_modules()
net1 = net1.to(device)
net2 = net2.to(device)
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

# # Training
# def train(epoch, net):
#     GREEN = '\033[92m'
#     RED = '\033[93m'
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     # freeze = True if epoch < args.freeze_before else False
#     freeze = False
#     color = GREEN if freeze else RED
#     e = '\nEpoch: %d  ' % epoch
#     print(e + color + ('Freeze' if freeze else 'Not Freeze') + '\033[0m')
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         # outputs = net(inputs, freeze=freeze)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#     print('train acc:',correct/total*100)
#     if using_wanbd:
#         wandb.log({'train acc':correct/total*100})

# def test(epoch, net):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     print('test_loss:',test_loss)
#     print('test Acc:', correct/total*100)
#     if using_wanbd:
#         wandb.log({'test acc':correct/total*100})
#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('best..')
#         best_acc = acc
#     #     torch.save(net.state_dict(), "./checkpoints/" + str(args.model) + "-{}-best.pth".format(args.tactic))

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


def train(epoch, net):
    correct = [0 for _ in range(5)]
    predicted = [0 for _ in range(5)]
    global init
    net.train()
    sum_loss, total = 0.0, 0.0
    for i, data in enumerate(trainloader, 0):
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        ensemble = sum(outputs) / len(outputs)
        ensemble.detach_()

        loss = torch.FloatTensor([0.]).to(device)

        for output in outputs:
            loss += criterion(output, labels) * (1 - args.loss_coefficient)

            # for other in outputs:
            #     if other is not output:
            #         #   logits distillation
            #         loss += kl_distill(output, other) * args.loss_coefficient / (len(outputs) - 1)

        sum_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(labels.size(0))
        outputs.append(ensemble)

        for classifier_index in range(len(outputs)):
            _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
            correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
        if i % 80 == 79:
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
                  ' Ensemble: %.2f%%' % (epoch, (i + epoch * length), sum_loss / (i + 1),
                                         100 * correct[0] / total, 100 * correct[1] / total,
                                         100 * correct[2] / total, 100 * correct[3] / total,
                                         100 * correct[4] / total))
    # wandb.log({'train_acc': 100. * correct[4] / total, 'train_acc1': 100. * correct[0] / total,
    #            'train_acc4': 100. * correct[3] / total, 'train_loss': sum_loss})


def test(epoch, net):
    with torch.no_grad():
        correct = [0 for _ in range(5)]
        predicted = [0 for _ in range(5)]
        total = 0.0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            ensemble = sum(outputs) / len(outputs)
            outputs.append(ensemble)
            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
            total += float(labels.size(0))

        print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
              ' Ensemble: %.4f%%' % (100 * correct[0] / total, 100 * correct[1] / total,
                                     100 * correct[2] / total, 100 * correct[3] / total,
                                     100 * correct[4] / total))
        # wandb.log({'test_acc': 100. * correct[4] / total, 'test_acc1': 100. * correct[0] / total,
        #            'test_acc4': 100. * correct[3] / total})

        global best_single, best_acc
        if correct[4] / total > best_acc:
            best_acc = correct[4] / total
            print("Best Accuracy Updated: ", best_acc * 100)
            # torch.save(net.state_dict(), "./checkpoints/" + str(args.model) + ".pth")
        for i in range(4):
            if correct[i] / total > best_single:
                best_single = correct[i] / total
                print("Best Single Accuracy Updated: ", best_single * 100)
                torch.save(net.state_dict(), "./checkpoints/" + str(args.model) + ".pth")
    print()

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

kl_distill = DistillKL(args.temperature)

if __name__ == '__main__':
    net2.module.bn1 = net1.module.bn1
    net2.module.conv1 = net1.module.conv1
    net2.module.layer1[0] = net1.module.layer1
    for epoch in range(args.epoch):
        test(0, net2)
        input()
        train(epoch, net2)
        adjust_lr(epoch+2)
    print('Finished, best acc',best_acc)
    # test_tree(net1)