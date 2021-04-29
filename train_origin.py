import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet_liu import *
from autoaugment import CIFAR10Policy
from cutout import Cutout
import torch.backends.cudnn as cudnn
import wandb

cudnn.benchmark = True

# set seed for reproducibility
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--model', default="tree_resnet32", type=str, help="resnet18|resnet34|resnet50|resnet101|resnet152|"
                                                                  "wideresnet50|wideresnet101|resnext50|resnext101")
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100|cifar10")
# default 270 epoch
parser.add_argument('--epoch', default=270, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--dataset_path', default="data", type=str)

# parser.add_argument('--autoaugment', default=True, type=bool)
parser.add_argument('--autoaugment', default=False, type=bool)

parser.add_argument('--temperature', default=3.0, type=float)
parser.add_argument('--batchsize', default=128 * 2, type=int)
parser.add_argument('--init_lr', default=0.1, type=float)
args = parser.parse_args()
print(args)


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

net = CifarResNet32(num_class)

net.to(device)
net = torch.nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
kl_distill = DistillKL(args.temperature)
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)


def train(epoch):
    correct = 0
    if epoch in epoch_down:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    net.train()
    sum_loss, total = 0.0, 0.0
    for i, data in enumerate(trainloader, 0):
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs= net(inputs)

        #   compute loss
        loss = torch.FloatTensor([0.]).to(device)
        loss += criterion(outputs, labels)

        sum_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(labels.size(0))

        _, predicted= torch.max(outputs.data, 1)
        correct+= float(predicted.eq(labels.data).cpu().sum())
        if i % 80 == 79:
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f ' % (epoch, (i + epoch * length), sum_loss / (i + 1),100 * correct / total))
    wandb.log({'train_acc': 100. * correct / total,'train_loss': sum_loss})


def test(epoch):
    with torch.no_grad():
        correct = 0
        total = 0.0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs= net(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())
            total += float(labels.size(0))

        print('Test Set AccuracyAcc: 4/4: %.4f' % (100 * correct / total))
        wandb.log({'test_acc': 100. * correct / total})

        global best_single, best_acc
        if correct/ total > best_acc:
            best_acc = correct / total
            print("Best Accuracy Updated: ", best_acc * 100)
    # scheduler.step()
    # print('lr:', scheduler.get_last_lr())
    print()


if __name__ == "__main__":
    best_acc = 0
    best_single = 0
    wandb.init(project="distill")
    if args.autoaugment==False:
        args.epoch=200
        epoch_down = [60,120,180]
    else:
        epoch_down = [90, 160, 210,250]
    for epoch in range(args.epoch):
        train(epoch)
        test(epoch)
    print("Training Finished, TotalEPOCH=%d, Best Accuracy=%.4f, Best Single=%.4f" % (
    args.epoch, 100 * best_acc, 100 * best_single))
