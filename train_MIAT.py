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
# from models.resnet_new import ResNet18
from models.wideresnet_new import WideResNet

from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet
from compute_MI import compute_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR MI AT')

parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
# parser.add_argument('--lr-mi', type=float, default=1e-3, metavar='LR',
 #                    help='learning rate')
parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')

parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')

parser.add_argument('--warm-up', type=bool, default=True,
                    help='warm up the MI estimator')
parser.add_argument('--warm-epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
'''
parser.add_argument('--pretrain-model', default='./checkpoint/resnet_18/ori/best_model.pth',
                    help='directory of model for saving checkpoint')
'''
parser.add_argument('--pre-local-n', default='./checkpoint/resnet_18/MI_estimator/beta_final_l2/local_n_model.pth',
                    help='directory of model for saving checkpoint')
parser.add_argument('--pre-global-n', default='./checkpoint/resnet_18/MI_estimator/beta_final_l2/global_n_model.pth',
                    help='directory of model for saving checkpoint')
parser.add_argument('--pre-local-a', default='./checkpoint/resnet_18/MI_estimator/beta_final_l2/local_a_model.pth',
                    help='directory of model for saving checkpoint')
parser.add_argument('--pre-global-a', default='./checkpoint/resnet_18/MI_estimator/beta_final_l2/global_a_model.pth',
                    help='directory of model for saving checkpoint')

parser.add_argument('--va-mode', choices=['nce', 'fd', 'dv'], default='dv')
parser.add_argument('--va-fd-measure', default='JSD')
parser.add_argument('--va-hsize', type=int, default=2048)
parser.add_argument('--is_internal', type=bool, default=False)
parser.add_argument('--is_internal_last', type=bool, default=False)

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./checkpoint/wideresnet/MIAT_standard',
                    help='directory of model for saving checkpoint')
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--save-freq', default=2, type=int, metavar='N', help='save frequency')

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


def make_optimizer_and_schedule(model, lr):
    optimizer = Adam(model.parameters(), lr)
    schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90], gamma=0.1)
    return optimizer, schedule


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr

    if epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def craft_adversarial_example_pgd(model, x_natural, y, step_size=0.007, epsilon=0.031, perturb_steps=20,
                                  distance='l_inf'):
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
    model.train()
    local_n.eval()
    global_n.eval()


    # logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    loss_ce = F.cross_entropy(logits_adv, y)
    # loss_ce = 0.2 * F.cross_entropy(logits_nat, y) + 0.8 * F.cross_entropy(logits_adv, y)

    pesudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    if torch.nonzero(index).size(0) != 0:


        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_mea = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        loss_a = loss_a.sum()/torch.nonzero(index).size(0)

        loss_mi = loss_mea + 0.1 * loss_a

    else:
        loss_mi = 0.0

    loss_all = loss_ce + loss_mi

    if (i + 1) % args.print_freq == 0:
        print('select samples:' + str(torch.nonzero(index).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train target model. Natural MI: %.4f; Loss_ce: %.4f; Loss_all: %.4f'
              % (epoch, i + 1, 50000 // args.batch_size, loss_mi.item(), loss_ce.item(), loss_all.item()))

    return loss_all


def MI_loss(i, model, x_natural, y, x_adv, local_n, global_n, local_a, global_a, epoch):
    model.train()
    local_n.eval()
    global_n.eval()
    local_a.eval()
    global_a.eval()

    # logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    loss_ce = F.cross_entropy(logits_adv, y)
    # loss_ce = 0.2 * F.cross_entropy(logits_nat, y) + 0.8 * F.cross_entropy(logits_adv, y)

    pesudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    if torch.nonzero(index).size(0) != 0:


        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        # loss_a_all = loss_a
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        # loss_a_all = torch.tensor(0.1).cuda() * (loss_a_all - loss_a)
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_mi = loss_mea_n + loss_mea_a # + loss_a_all

    else:
        loss_mi = 0.0

    loss_all = loss_ce + 5.0 * loss_mi

    if (i + 1) % args.print_freq == 0:
        print('select samples:' + str(torch.nonzero(index).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train target model. Natural MI: %.4f; Loss_ce: %.4f; Loss_all: %.4f'
              % (epoch, i + 1, 50000 // args.batch_size, loss_mi.item(), loss_ce.item(), loss_all.item()))

    return loss_all


def evaluate_mi_nat(encoder, x_natural, y, x_adv, local_n, global_n):

    encoder.eval()
    local_n.eval()
    global_n.eval()

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label == y)

    loss_r_n = (compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_r_a = (compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    loss_w_n = (compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_w_a = (compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    return loss_r_n, loss_r_a, loss_w_n, loss_w_a


def evaluate_mi_adv(encoder, x_natural, y, x_adv, local_n, global_n):

    encoder.eval()
    local_n.eval()
    global_n.eval()

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label == y)

    loss_r_n = (compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_r_a = (compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    loss_w_n = (compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_w_a = (compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    return loss_r_n, loss_r_a, loss_w_n, loss_w_a


def eval_test(model, device, test_loader, local_n, global_n, local_a, global_a):
    model.eval()
    local_n.eval()
    global_n.eval()
    local_a.eval()
    global_a.eval()

    cnt = 0
    correct = 0
    correct_adv = 0
    losses_r_n = 0
    losses_r_a = 0
    losses_w_n = 0
    losses_w_a = 0
    losses_r_n_1 = 0
    losses_r_a_1 = 0
    losses_w_n_1 = 0
    losses_w_a_1 = 0

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

            test_loss_r_n, test_loss_r_a, test_loss_w_n, test_loss_w_a = evaluate_mi_nat(encoder=model, x_natural=data,
                                                    y=target, x_adv=data_adv, local_n=local_n, global_n=global_n)

            test_loss_r_n_1, test_loss_r_a_1, test_loss_w_n_1, test_loss_w_a_1 = evaluate_mi_nat(encoder=model, x_natural=data,
                                                                                     y=target, x_adv=data_adv,
                                                                                     local_n=local_a, global_n=global_a)

        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        losses_r_n += test_loss_r_n.item()
        losses_r_a += test_loss_r_a.item()
        losses_w_n += test_loss_w_n.item()
        losses_w_a += test_loss_w_a.item()

        losses_r_n_1 += test_loss_r_n_1.item()
        losses_r_a_1 += test_loss_r_a_1.item()
        losses_w_n_1 += test_loss_w_n_1.item()
        losses_w_a_1 += test_loss_w_a_1.item()

    test_accuracy = (correct_adv + correct) / (2.0 * len(test_loader.dataset))
    print('Test:  Accuracy: {}/{} ({:.2f}%), Robust Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
        100. * correct_adv / len(test_loader.dataset)))
    print('Test: Natural MI Right: -n: {:.4f}, -a: {:.4f}'.format(
        losses_r_n/cnt, losses_r_a/cnt))
    print('Test: Natural MI Wrong: -n: {:.4f}, -a: {:.4f}'.format(
        losses_w_n / cnt, losses_w_a / cnt))
    print('Test: Adv MI Right: -n: {:.4f}, -a: {:.4f}'.format(
        losses_r_n_1/cnt, losses_r_a_1/cnt))
    print('Test: Adv MI Wrong: -n: {:.4f}, -a: {:.4f}'.format(
        losses_w_n_1 / cnt, losses_w_a_1 / cnt))

    return test_accuracy



def main():
    # settings
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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

    # Estimator part 1: X or layer3 to H space
    local_n = Estimator(args.va_hsize)
    local_a = Estimator(args.va_hsize)

    # estimator part 2: Z to H space
    if args.is_internal == True:
        if args.is_internal_last == True:
            z_size = 512
            global_n = MIInternallastConvNet(z_size, args.va_hsize)
            global_a = MIInternallastConvNet(z_size, args.va_hsize)
        else:
            z_size = 256
            global_n = MIInternalConvNet(z_size, args.va_hsize)
            global_a = MIInternalConvNet(z_size, args.va_hsize)
    else:
        z_size = 10
        global_n = MI1x1ConvNet(z_size, args.va_hsize)
        global_a = MI1x1ConvNet(z_size, args.va_hsize)

    # target_model = ResNet18(10)
    target_model = WideResNet(34, 10, 10)
    target_model = torch.nn.DataParallel(target_model).cuda()

    local_n.load_state_dict(torch.load(args.pre_local_n))
    global_n.load_state_dict(torch.load(args.pre_global_n))
    local_a.load_state_dict(torch.load(args.pre_local_a))
    global_a.load_state_dict(torch.load(args.pre_global_a))

    local_n = torch.nn.DataParallel(local_n).cuda()
    global_n = torch.nn.DataParallel(global_n).cuda()
    local_a = torch.nn.DataParallel(local_a).cuda()
    global_a = torch.nn.DataParallel(global_a).cuda()

    cudnn.benchmark = True

    optimizer = optim.SGD(target_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # opt_local_n, schedule_local_n = make_optimizer_and_schedule(local_n, lr=args.lr_mi)
    # opt_global_n, schedule_global_n = make_optimizer_and_schedule(global_n, lr=args.lr_mi)
    # opt_local_a, schedule_local_a = make_optimizer_and_schedule(local_a, lr=args.lr_mi)
    # opt_global_a, schedule_global_a = make_optimizer_and_schedule(global_a, lr=args.lr_mi)

    # warm up
    print('--------Warm up--------')
    for epocah in range(0, 2):
        for batch_idx, (data, target) in enumerate(train_loader):
            target_model.train()

            data, target = data.to(device), target.to(device)

            logits_nat = target_model(data)

            loss = F.cross_entropy(logits_nat, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # Train
    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)

        print('--------Train the target model--------')

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # craft adversarial examples
            adv = craft_adversarial_example_pgd(model=target_model, x_natural=data, y=target, step_size=0.007,
                                                epsilon=8/255, perturb_steps=40, distance='l_inf')

            # Train MI estimator
            loss = MI_loss(i=batch_idx, model=target_model, x_natural=data, y=target, x_adv=adv, local_n=local_n,
                           global_n=global_n, local_a=local_a, global_a=global_a, epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluation
        print('--------Evaluate the target model--------')

        test_accuracy = eval_test(model=target_model, device=device, test_loader=test_loader, local_n=local_n,
                                  global_n=global_n, local_a=local_a, global_a=global_a)

        # save checkpoint
        if test_accuracy >= best_accuracy:  # epoch % args.save_freq == 0:
            best_accuracy = test_accuracy
            '''
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch)))
            '''

            torch.save(target_model.module.state_dict(),
                       os.path.join(args.model_dir, 'target_model.pth'))
            '''
            torch.save(local_n.module.state_dict(),
                       os.path.join(args.model_dir, 'local_model.pth'))
            torch.save(global_n.module.state_dict(),
                       os.path.join(args.model_dir, 'global_model.pth'))
            '''
            print('save the model')

        print('================================================================')


if __name__ == '__main__':
    main()
