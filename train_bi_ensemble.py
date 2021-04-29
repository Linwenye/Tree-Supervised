import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from models.resnet_liu import *
import torch.nn.functional as F
from utils.autoaugment import CIFAR10Policy
from utils.cutout import Cutout
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
GPU_double = 2

# set seed for reproducibility
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--model', default="resnet18", type=str, help="resnet18|resnet34|resnet50|resnet101|resnet152|"
                                                                  "wideresnet50|wideresnet101|resnext50|resnext101")
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100|cifar10")
# default 250 epoch
parser.add_argument('--epoch', default=250, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--dataset_path', default="data", type=str)

parser.add_argument('--autoaugment', default=True, type=bool)
# parser.add_argument('--autoaugment', default=False, type=bool)

parser.add_argument('--temperature', default=3.0, type=float)
parser.add_argument('--batchsize', default=128 * GPU_double, type=int)
parser.add_argument('--init_lr', default=0.1, type=float)
args = parser.parse_args()
print(args)


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
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=4
)

if args.model == "resnet18":
    net = BiResNet18(100)
if args.model == "resnet34":
    net = resnet34()
if args.model == "resnet50":
    net = resnet50()
if args.model == "resnet101":
    net = resnet101()
if args.model == "resnet152":
    net = resnet152()
if args.model == "wideresnet50":
    net = wide_resnet50_2()
if args.model == "wideresnet101":
    net = wide_resnet101_2()
if args.model == "resnext50_32x4d":
    net = resnet18()
if args.model == "resnext101_32x8d":
    net = resnext101_32x8d()

net.to(device)
net = torch.nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
kl_distill = DistillKL(args.temperature)
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)


# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    correct = [0 for _ in range(5)]
    predicted = [0 for _ in range(5)]
    if epoch in [90, 160, 210, 250]:

        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    net.train()
    sum_loss, total = 0.0, 0.0
    for i, data in enumerate(trainloader, 0):
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, outputs_feature = net(inputs)
        ensemble = sum(outputs) / len(outputs)
        ensemble.detach_()

        #   compute loss
        loss = torch.FloatTensor([0.]).to(device)

        # using out1 and out4 as teacher per epoch
        #   teacher: -temp: swap; -temp: out4; -further: random; -further: mutual
        # further er : distill by ensemble
        for index in range(len(outputs)):
            #   logits distillation
            loss += kl_distill(outputs[index], ensemble) * args.loss_coefficient

            # loss += criterion(outputs[index], labels)  change the parameters to see the result
            loss += criterion(outputs[index], labels) * (1 - args.loss_coefficient)

        sum_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(labels.size(0))
        outputs.append(ensemble)

        for classifier_index in range(len(outputs)):
            _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
            correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
    #     if i % 80 == 79:
    #         print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
    #               ' Ensemble: %.2f%%' % (epoch, (i + epoch * length), sum_loss / (i + 1),
    #                                      100 * correct[0] / total, 100 * correct[1] / total,
    #                                      100 * correct[2] / total, 100 * correct[3] / total,
    #                                      100 * correct[4] / total))
    # wandb.log({'train_acc': 100. * correct[4] / total, 'train_acc1': 100. * correct[0] / total,
    #            'train_acc4': 100. * correct[3] / total, 'train_loss': sum_loss})


def test(epoch):
    with torch.no_grad():
        correct = [0 for _ in range(5)]
        predicted = [0 for _ in range(5)]
        total = 0.0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, outputs_feature = net(images)
            ensemble = sum(outputs) / len(outputs)
            outputs.append(ensemble)
            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
            total += float(labels.size(0))

        # print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
        #       ' Ensemble: %.4f%%' % (100 * correct[0] / total, 100 * correct[1] / total,
        #                              100 * correct[2] / total, 100 * correct[3] / total,
        #                              100 * correct[4] / total))
        # wandb.log({'test_acc': 100. * correct[4] / total, 'test_acc1': 100. * correct[0] / total,
        #            'test_acc4': 100. * correct[3] / total})

        global best_single, best_acc
        if correct[4] / total > best_acc:
            best_acc = correct[4] / total
            # print("Best Accuracy Updated: ", best_acc * 100)
            # torch.save(net.state_dict(), "./checkpoints/" + str(args.model) + ".pth")
        for i in range(4):
            if correct[i] / total > best_single:
                best_single = correct[i] / total
                # print("Best Single Accuracy Updated: ", best_single * 100)
                # torch.save(net.state_dict(), "./checkpoints/" + str(args.model) + ".pth")
    # scheduler.step()
    # print('lr:', scheduler.get_last_lr())


if __name__ == "__main__":
    best_acc = 0
    best_single = 0
    # wandb.init(project="distill")
    for epoch in range(args.epoch):
        train(epoch)
        test(epoch)
    print("Training Finished, TotalEPOCH=%d, Best Accuracy=%.4f, Best Single=%.4f" % (
        args.epoch, 100 * best_acc, 100 * best_single))
    print()