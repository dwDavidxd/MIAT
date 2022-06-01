from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import torch.optim as optim
from torchvision import transforms

from models.resnet_combine_new import ResNet18
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
parser.add_argument('--model-dir', default='./checkpoint/resnet_18/MAN_adaptive/standard_bs_128',
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

    if epoch >= 100:  # 90
        lr = args.lr * 0.001
    elif epoch >= 90:  # 55
        lr = args.lr * 0.01
    elif epoch >= 75:  # 35
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr

    if epoch >= 140:  # 90
        lr = args.lr * 0.01
    elif epoch >= 130:  # 90
        lr = args.lr * 0.1
    elif epoch >= 90 :  # 55
        lr = args.lr * 0.01
    elif epoch >= 75:  # 35
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''


def craft_adversarial_example(model, x_natural, y, step_size=0.003,
                epsilon=0.031, perturb_steps=10, distance='l_inf'):

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():

                noisy_post = model(x_adv)
                log_noisy_post = torch.log(noisy_post + 1e-12)
                loss = nn.NLLLoss()(log_noisy_post, y)

            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def standard_loss(model, x_natural, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf'):

    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                noisy_post = model(x_adv)
                log_noisy_post = torch.log(noisy_post + 1e-12)
                loss_ce = nn.NLLLoss()(log_noisy_post, y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    '''
    x_mixture = torch.cat((x_natural, x_adv), 0)
    y_mixture = torch.cat((y, y), 0)
    optimizer.zero_grad()
    T = model(x_mixture, out_T=True)
    logits_ori = model(x_mixture, out_ori=True).detach()
    pred_labels = F.softmax(logits_ori, dim=1)
    natural_post = torch.bmm(pred_labels.unsqueeze(1), T).squeeze(1)  # softmax output   
    loss_adv = F.cross_entropy(natural_post, y_mixture)

    loss_adv.backward()
    optimizer.step()

    optimizer.zero_grad()
    T = model(x_mixture, out_T=True).detach()
    logits_ori = model(x_mixture, out_ori=True)
    pred_labels = F.softmax(logits_ori, dim=1)
    natural_post = torch.bmm(pred_labels.unsqueeze(1), T).squeeze(1)  # softmax output   
    loss_adv = F.cross_entropy(natural_post, y_mixture)# + F.cross_entropy(logits_ori[0:x_natural.size(0)], y)

    loss_adv.backward()
    optimizer.step()
    '''
    optimizer.zero_grad()

    noisy_post = model(x_adv)
    logits_adv = torch.log(noisy_post + 1e-12)

    noisy_post = model(x_natural)
    logits_nat = torch.log(noisy_post + 1e-12)

    loss_adv = 1.0 * nn.NLLLoss()(logits_adv, y) + 0.0 * nn.NLLLoss()(logits_nat, y)

    loss_adv.backward()
    optimizer.step()

    return loss_adv


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss

        loss = standard_loss(model=model, x_natural=data, y=target, optimizer=optimizer,
                                        step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps,
                                        distance='l_inf')

        # print progress
        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct_adv = 0
    correct_adv_ori = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_adv = craft_adversarial_example(model=model, x_natural=data, y=target,
                                                 step_size=0.007, epsilon=8/255,
                                                 perturb_steps=40, distance='l_inf')

            output = model(data)
            output_adv, output_adv_ori = model(data_adv, out_all=True)

            pred = output.max(1, keepdim=True)[1]
            pred_adv = output_adv.max(1, keepdim=True)[1]
            pred_adv_ori = output_adv_ori.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
            correct_adv_ori += pred_adv_ori.eq(target.view_as(pred_adv_ori)).sum().item()

        print('Test: Accuracy: {}/{} ({:.0f}%), Robust Accuracy: {}/{} ({:.0f}%), Robust ori Accuracy:'
              ' {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
            100. * correct_adv / len(test_loader.dataset), correct_adv_ori, len(test_loader.dataset),
            100. * correct_adv_ori / len(test_loader.dataset)))

        test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main():
    # settings
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
    model = ResNet18(10, 100).to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # best_pred = 0

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        # start = perf_counter()

        train(args, model, device, train_loader, optimizer, epoch)

        # end = perf_counter()

        #print(f"Time taken to execute code : {end - start}")

        # evaluation on natural examples
        print('================================================================')
        # eval_train(model, device, train_loader)
        _, test_accuracy = eval_test(model, device, test_loader)

        # save checkpoint
        if epoch % args.save_freq == 0:

            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'best_model.pth'))
            print('save the model')

        print('================================================================')


if __name__ == '__main__':
    main()
