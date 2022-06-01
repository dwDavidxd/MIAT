from __future__ import print_function
import os
import argparse
import torch
# import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

# from models.wideresnet import *
from models.resnet import ResNet18
from utils.standard_loss import standard_loss
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import data_dataset
import numpy as np
from time import perf_counter, sleep

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')

parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=90, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/resnet_18/Standard_bs_128',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=90, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr

    if epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss

        loss = standard_loss(model=model, x_natural=data, y=target, optimizer=optimizer, step_size=args.step_size,
                             epsilon=args.epsilon, perturb_steps=args.num_steps, distance='l_inf')

        # loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))
            '''
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
            '''


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def craft_adversarial_example(model, x_natural, y, step_size=0.003,
                epsilon=0.031, perturb_steps=10, distance='l_inf'):

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_kl = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        batch_size = len(x_natural)
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.cross_entropy(model(adv), y)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct_adv = 0
    #with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_adv = craft_adversarial_example(model=model, x_natural=data, y=target,
                                             step_size=0.007, epsilon=8/255,
                                             perturb_steps=40, distance='l_inf')

        output = model(data)
        output_adv = model(data_adv)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        pred_adv = output_adv.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Robust Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
        100. * correct_adv / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main():
    # settings
    setup_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # setup data loader
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = data_dataset(img_path=args.nat_img_train, clean_label_path=args.nat_label_train,
                            transform=trans_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=True, num_workers=4, pin_memory=True)
    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

    # init model, ResNet18() can be also used here for training
    model = ResNet18(10).to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        # start = perf_counter()

        train(args, model, device, train_loader, optimizer, epoch)

        # end = perf_counter()

        # print(f"Time taken to execute code : {end - start}")

        # evaluation on natural examples
        print('================================================================')
        # eval_train(model, device, train_loader)
        _, test_accuracy = eval_test(model, device, test_loader)

        # save checkpoint
        if epoch % args.save_freq == 0:
            '''
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch)))
            '''

            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'best_model.pth'))
            print('save the model')

        print('================================================================')


if __name__ == '__main__':
    main()
