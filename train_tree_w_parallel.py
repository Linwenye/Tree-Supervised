import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from models import *
import torch.nn.functional as F
from utils.autoaugment import CIFAR10Policy
from utils.cutout import Cutout
import torch.backends.cudnn as cudnn
import torch
from torch import nn
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time


parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--model', default="tree_resnet32", type=str, help="resnet18|tree_resnet32|tree_wide|tree_mobilev3|mobilev3|wide")
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100|cifar10")
parser.add_argument('--epoch', default=200, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--dataset_path', default="data", type=str)

# parser.add_argument('--autoaugment', default=True, type=bool)
parser.add_argument('--autoaugment', default=False, type=bool)

parser.add_argument('--temperature', default=3.0, type=float)
parser.add_argument('--batchsize', default=128 * 2, type=int)
parser.add_argument('--init_lr', default=0.1, type=float)
args = parser.parse_args()


# print(args)


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


cudnn.benchmark = True
# set seed for reproducibility
best_acc = 0
best_single = 0


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')


def cleanup():
    dist.destroy_process_group()


if args.autoaugment:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                          transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                                          Cutout(n_holes=1, length=16),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
else:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def train(rank, world_size):
    torch.manual_seed(rank+1)

    if rank == 0:
        print(args)
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    args.batchsize = int(args.batchsize / world_size)

    # ------------------------ load data
    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=args.dataset_path,
            train=True,
            download=True,
            transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=args.dataset_path,
            train=False,
            download=True,
            transform=transform_test
        )
        num_class = 100
    elif args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=args.dataset_path,
            train=True,
            download=True,
            transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=args.dataset_path,
            train=False,
            download=True,
            transform=transform_test
        )
        num_class = 10
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    # -------------------------------------
    if args.model == 'tree_wide':
        net = Wide_TreeResNet(28, 10, 0, num_class)
    elif args.model =='tree_mobilev3':
        net = TreeMobileNetV3_Large(num_class)
    elif args.model == 'mobilev3':
        net = MobileNetV3_Large(num_class)
    elif args.model == 'wide':
        net = Wide_ResNet(28, 10, 0, num_class)
    else:
        raise NameError

    # create model and move it to GPU with id rank

    net = net.to(rank)
    net = DDP(net, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    kl_distill = DistillKL(args.temperature)
    optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)

    optimizer.zero_grad()

    for epoch in range(args.epoch):
        ######################### train
        train_start = time.time()
        correct = [0 for _ in range(5)]
        predicted = [0 for _ in range(5)]
        if epoch in [60, 120, 180]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss, total = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(rank), labels.to(rank)
            outputs = net(inputs)
            ensemble = sum(outputs) / len(outputs)
            ensemble.detach_()

            #   compute loss
            loss = torch.FloatTensor([0.]).to(rank)

            for output in outputs:
                loss += criterion(output, labels) * (1 - args.loss_coefficient)

                for other in outputs:
                    if other is not output:
                        #   logits distillation
                        loss += kl_distill(output, other) * args.loss_coefficient / (len(outputs) - 1)

            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(labels.size(0))
            outputs.append(ensemble)

            if rank == 0:
                for classifier_index in range(len(outputs)):
                    _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                    correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
                if i % 80 == 79:
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
                          ' Ensemble: %.2f%%' % (epoch, (i + epoch * length), sum_loss / (i + 1),
                                                 100 * correct[0] / total, 100 * correct[1] / total,
                                                 100 * correct[2] / total, 100 * correct[3] / total,
                                                 100 * correct[4] / total))
        if rank == 0:
            print('train epoch time:',time.time()-train_start)
            print({'train_acc': 100. * correct[4] / total, 'train_acc1': 100. * correct[0] / total,
                   'train_acc4': 100. * correct[3] / total, 'train_loss': sum_loss})

        ################################# test
        if rank == 3:
            with torch.no_grad():
                correct = [0 for _ in range(5)]
                predicted = [0 for _ in range(5)]
                total = 0.0
                net.eval()
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(rank), labels.to(rank)
                    outputs = net(images)
                    ensemble = sum(outputs) / len(outputs)
                    outputs.append(ensemble)
                    for classifier_index in range(len(outputs)):
                        _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                        correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
                    total += float(labels.size(0))
                print('train and test:',time.time()-train_start)
                print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
                      ' Ensemble: %.4f%%' % (100 * correct[0] / total, 100 * correct[1] / total,
                                             100 * correct[2] / total, 100 * correct[3] / total,
                                             100 * correct[4] / total))
                print({'test_acc': 100. * correct[4] / total, 'test_acc1': 100. * correct[0] / total,
                       'test_acc4': 100. * correct[3] / total})

                global best_single, best_acc
                if correct[4] / total > best_acc:
                    best_acc = correct[4] / total
                    print("Best Accuracy Updated: ", best_acc * 100)
                    # torch.save(net.state_dict(), "./checkpoints/" + str(args.model) + ".pth")
                for i in range(4):
                    if correct[i] / total > best_single:
                        best_single = correct[i] / total
                        print("Best Single Accuracy Updated: ", best_single * 100)
                        # torch.save(net.state_dict(), "./checkpoints/" + str(args.model) + ".pth")
                print()
    if rank == 0:
        print("Training Finished, TotalEPOCH=%d, Best Accuracy=%.4f, Best Single=%.4f" % (
            args.epoch, 100 * best_acc, 100 * best_single))
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    run_demo(train, 4)
