# This version max Natural MI of x and max Adversarial MI of x_adv

import os
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler, Adam
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from data import data_dataset
from models.resnet_new import ResNet18

from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet
from compute_MI import compute_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR MI AT')

parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr-mi', type=float, default=1e-2, metavar='LR',
                    help='learning rate')

parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')

parser.add_argument('--pre-target', default='./checkpoint/resnet_18/standard_AT_pre/best_model.pth',
                    help='directory of model for saving checkpoint')

parser.add_argument('--va-mode', choices=['nce', 'fd', 'dv'], default='dv')
parser.add_argument('--va-fd-measure', default='JSD')
parser.add_argument('--va-hsize', type=int, default=2048)
parser.add_argument('--is_internal', type=bool, default=False)
parser.add_argument('--is_internal_last', type=bool, default=False)

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./checkpoint/resnet_18/MI_estimator/only_adv_max',
                    help='directory of model for saving checkpoint')
parser.add_argument('--print_freq', default=50, type=int)
parser.add_argument('--save-freq', default=1, type=int, metavar='N', help='save frequency')

args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def make_optimizer(model, lr):
    optimizer = Adam(model.parameters(), lr)
    return optimizer


def make_optimizer_and_schedule(model, lr):
    optimizer = Adam(model.parameters(), lr)
    schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)
    return optimizer, schedule


def craft_adversarial_example_pgd(model, x_natural, y, step_size=0.007,
                epsilon=0.031, perturb_steps=20, distance='l_inf'):
    model.eval()

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_ce = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
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


def MI_loss_nat(i, model, x_natural, y, x_adv, local_n, global_n, epoch):
    model.eval()
    local_n.train()
    global_n.train()

    loss_n = compute_loss(args=args, former_input=x_adv, latter_input=x_adv, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True).mean()

    if (i + 1) % args.print_freq == 0:

        print('Epoch [%d], Iter [%d/%d] Train MI estimator. Natural MI: -n %.4f'
              % (epoch, i + 1, 50000 // args.batch_size, loss_n.item()))

    return loss_n


def evaluate_mi_nat(encoder, x_natural, y, x_adv, local_n, global_n):
    encoder.eval()
    local_n.eval()
    global_n.eval()

    loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True).mean()
    loss_a = compute_loss(args=args, former_input=x_adv, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True).mean()
    return loss_n, loss_a


def eval_test(model, device, test_loader, local_n, global_n):
    model.eval()
    local_n.eval()
    global_n.eval()


    cnt = 0
    correct = 0
    correct_adv = 0
    losses_n_n = 0
    losses_n_a = 0

    for data, target in test_loader:
        cnt += 1
        data, target = data.to(device), target.to(device)
        data_adv = craft_adversarial_example_pgd(model=model, x_natural=data, y=target,
                                             step_size=0.007, epsilon=8/255,
                                             perturb_steps=40, distance='l_inf')

        with torch.no_grad():
            output = model(data)
            output_adv = model(data_adv)
            pred = output.max(1, keepdim=True)[1]
            pred_adv = output_adv.max(1, keepdim=True)[1]

            test_loss_n_n, test_loss_n_a = evaluate_mi_nat(encoder=model, x_natural=data, y=target, x_adv=data_adv,
                                                           local_n=local_n, global_n=global_n)

        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        losses_n_n += test_loss_n_n.item()
        losses_n_a += test_loss_n_a.item()


    test_accuracy = correct_adv / len(test_loader.dataset)
    print('Test:  Accuracy: {}/{} ({:.2f}%), Robust Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
        100. * correct_adv / len(test_loader.dataset)))
    print('Test: Natural MI: -n: {:.4f}, -a: {:.4f}'.format(
        losses_n_n/cnt, losses_n_a/cnt))

    return test_accuracy


def main():
    # settings
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    device = torch.device("cuda")

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

    # load MI estimation model

    # Estimator part 1: X or layer3 to H space
    local_n = Estimator(args.va_hsize)


    # estimator part 2: Z to H space
    if args.is_internal == True:
        if args.is_internal_last == True:
            z_size = 512
            global_n = MIInternallastConvNet(z_size, args.va_hsize)

        else:
            z_size = 256
            global_n = MIInternalConvNet(z_size, args.va_hsize)

    else:
        z_size = 10
        global_n = MI1x1ConvNet(z_size, args.va_hsize)


    print('----------------Start training-------------')
    target_model = ResNet18(10)

    state_dic = torch.load(args.pre_target)
    new_state = target_model.state_dict()

    for k in state_dic.keys():
        if k in new_state.keys():
            new_state[k] = state_dic[k]
            # print(k)
        else:
            break

    target_model.load_state_dict(new_state)

    target_model = torch.nn.DataParallel(target_model).cuda()

    local_n = torch.nn.DataParallel(local_n).cuda()
    global_n = torch.nn.DataParallel(global_n).cuda()


    cudnn.benchmark = True

    opt_local_n, schedule_local_n = make_optimizer_and_schedule(local_n, lr=args.lr_mi)
    opt_global_n, schedule_global_n = make_optimizer_and_schedule(global_n, lr=args.lr_mi)


    # Train
    for epoch in range(1, args.epochs + 1):
        loss_n_all = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # craft adversarial examples
            adv = craft_adversarial_example_pgd(model=target_model, x_natural=data, y=target)

            # Train MI estimator
            loss_n = MI_loss_nat(i=batch_idx, model=target_model, x_natural=data, y=target, x_adv=adv,
                           local_n=local_n, global_n=global_n, epoch=epoch)


            opt_local_n.zero_grad()
            opt_global_n.zero_grad()
            loss_n.backward()
            opt_local_n.step()
            opt_global_n.step()

        schedule_local_n.step()
        schedule_global_n.step()

        # evaluation
        print('================================================================')
        # _ = eval_train(model=target_model, device=device, test_loader=train_loader, local_n=local_n,
        #              global_n=global_n)

        test_accuracy = eval_test(model=target_model, device=device, test_loader=test_loader, local_n=local_n,
                                  global_n=global_n)

        # save checkpoint
        if epoch % args.save_freq == 0:

            '''
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch)))
            '''

            torch.save(local_n.module.state_dict(),
                       os.path.join(args.model_dir, 'local_n_model.pth'))
            torch.save(global_n.module.state_dict(),
                       os.path.join(args.model_dir, 'global_n_model.pth'))
            print('save the model')

        print('================================================================')


if __name__ == '__main__':
    main()
