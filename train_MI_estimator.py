# This version use cosine distance to enhance the difference between the MI of adv and the MI of nat.
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

parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr-mi', type=float, default=1e-2, metavar='LR',
                    help='learning rate')

parser.add_argument('--epsilon', default=0.5,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')

parser.add_argument('--pre-target', default='./checkpoint/resnet_18/standard_AT_pre_l2/best_model.pth',
                    help='directory of model for saving checkpoint')

parser.add_argument('--va-mode', choices=['nce', 'fd', 'dv'], default='dv')
parser.add_argument('--va-fd-measure', default='JSD')
parser.add_argument('--va-hsize', type=int, default=2048)
parser.add_argument('--is_internal', type=bool, default=False)
parser.add_argument('--is_internal_last', type=bool, default=False)

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./checkpoint/resnet_18/MI_estimator/beta_final_l2',
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
    schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
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

    pesudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    index_s = index

    if torch.nonzero(index).size(0) != 0:

        pesudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
        index_s = index * (pesudo_label == y)
        loss_n = 0

        if torch.nonzero(index_s).size(0) != 0:

            loss_n = (compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                                   dim_local=local_n, dim_global=global_n, v_out=True) * index_s).sum() / torch.nonzero(
                index).size(0)

            loss_n_s = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                                    dim_local=local_n, dim_global=global_n, v_out=True) * index_s
            loss_a_s = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                                    dim_local=local_n, dim_global=global_n, v_out=True) * index_s
            loss_mea = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n_s, loss_a_s, dim=0))

            loss_r = 5.0 * loss_mea + loss_n
        else:
            loss_r = 0

        pesudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
        index_s = index * (pesudo_label != y)

        if torch.nonzero(index_s).size(0) != 0:

            loss_n = (compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                                   dim_local=local_n, dim_global=global_n, v_out=True) * index_s).sum() / torch.nonzero(
                index).size(0)
            '''
            loss_a = (compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                                   dim_local=local_n, dim_global=global_n, v_out=True) * index_s).sum() / torch.nonzero(
                index).size(0)
            '''

            loss_n_s = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                    dim_local=local_n, dim_global=global_n, v_out=True) * index_s
            loss_a_s = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                                   dim_local=local_n, dim_global=global_n, v_out=True) * index_s
            loss_mea = torch.abs(torch.cosine_similarity(loss_n_s, loss_a_s, dim=0))


            # loss_w = loss_n - 0.15 * loss_a
            loss_w = 5.0 * loss_mea + loss_n # - 0.5 * loss_a
        else:
            loss_w = 0


        loss_all = 1.0 * loss_w + 0.5 * loss_r # 3

    else:
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                                dim_local=local_n, dim_global=global_n, v_out=True).mean()

        loss_all = loss_n

    if (i + 1) % args.print_freq == 0:
        print('select right nat samples; wrong adv samples:' + str(torch.nonzero(index).size(0)) + ';' + str(torch.nonzero(index_s).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train MI estimator. Natural MI: -n %.4f; Loss: -n %.4f'
              % (epoch, i + 1, 50000 // args.batch_size, loss_n.item(), loss_all.item()))

    return loss_all


def MI_loss_adv(i, model, x_natural, y, x_adv, local_n, global_n, epoch):
    model.eval()
    local_n.train()
    global_n.train()

    pesudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    index_s = index
    loss_a = 0

    if torch.nonzero(index).size(0) != 0:

        pesudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
        index_s = index * (pesudo_label != y)

        if torch.nonzero(index_s).size(0) != 0:

            loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=model,
                                   dim_local=local_n, dim_global=global_n, v_out=True) * index_s

            loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=model,
                                   dim_local=local_n, dim_global=global_n, v_out=True) * index_s

            loss_mea = torch.abs(torch.cosine_similarity(loss_a, loss_n, dim=0))
            loss_a = loss_a.sum()/torch.nonzero(index_s).size(0)
            loss_all_w = 5.0 * loss_mea + loss_a #5

        else:
            loss_all_w = 0

        pesudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
        index_s = index * (pesudo_label == y)

        if torch.nonzero(index_s).size(0) != 0:
            loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=model,
                                  dim_local=local_n, dim_global=global_n, v_out=True) * index_s

            loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=model,
                                  dim_local=local_n, dim_global=global_n, v_out=True) * index_s

            loss_mea = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))
            loss_a = loss_a.sum() / torch.nonzero(index_s).size(0)
            loss_all_r = 8.0 * loss_mea - 0.1 * loss_a  # 5
        else:
            loss_all_r = 0

        loss_all = 1.0 * loss_all_w + 0.5 * loss_all_r

    else:
        loss_a = compute_loss(args=args, former_input=x_adv-x_natural, latter_input=x_adv, encoder=model,
                                dim_local=local_n, dim_global=global_n, v_out=True).mean()

        loss_all = loss_a

    if (i + 1) % args.print_freq == 0:
        print('select right natural samples; right adv samples:' + str(torch.nonzero(index).size(0)) + ';' + str(torch.nonzero(index_s).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train MI estimator. Adversasrial MI: -n %.4f; Loss: -n %.4f'
              % (epoch, i + 1, 50000 // args.batch_size, loss_a.item(), loss_all.item()))

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

    loss_r_n = (compute_loss(args=args, former_input=x_adv-x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_r_a = (compute_loss(args=args, former_input=x_adv-x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)


    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    loss_w_n = (compute_loss(args=args, former_input=x_adv-x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_w_a = (compute_loss(args=args, former_input=x_adv-x_natural, latter_input=x_adv, encoder=encoder,
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
    losses_n_n_r = 0
    losses_n_a_r = 0
    losses_n_n_w = 0
    losses_n_a_w = 0
    losses_a_n_r = 0
    losses_a_a_r = 0
    losses_a_n_w = 0
    losses_a_a_w = 0


    for data, target in test_loader:
        cnt += 1
        data, target = data.to(device), target.to(device)
        data_adv = craft_adversarial_example_pgd(model=model, x_natural=data, y=target,
                                             step_size=0.007, epsilon=0.5,
                                             perturb_steps=40, distance='l_2')

        with torch.no_grad():
            output = model(data)
            output_adv = model(data_adv)
            pred = output.max(1, keepdim=True)[1]
            pred_adv = output_adv.max(1, keepdim=True)[1]

            test_loss_n_n_r, test_loss_n_a_r, test_loss_n_n_w, test_loss_n_a_w = evaluate_mi_nat(encoder=model, x_natural=data, y=target, x_adv=data_adv,
                                                           local_n=local_n, global_n=global_n)

            test_loss_a_n_r, test_loss_a_a_r, test_loss_a_n_w, test_loss_a_a_w = evaluate_mi_adv(encoder=model, x_natural=data, y=target, x_adv=data_adv,
                                                           local_n=local_a, global_n=global_a)


        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        losses_n_n_r += test_loss_n_n_r.item()
        losses_n_a_r += test_loss_n_a_r.item()
        losses_n_n_w += test_loss_n_n_w.item()
        losses_n_a_w += test_loss_n_a_w.item()
        losses_a_n_r += test_loss_a_n_r.item()
        losses_a_a_r += test_loss_a_a_r.item()
        losses_a_n_w += test_loss_a_n_w.item()
        losses_a_a_w += test_loss_a_a_w.item()

    test_accuracy = correct_adv / len(test_loader.dataset)
    print('Test:  Accuracy: {}/{} ({:.2f}%), Robust Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
        100. * correct_adv / len(test_loader.dataset)))
    print('Test: Natural MI: Right samples: -n: {:.4f}, -a: {:.4f}; Wrong samples: -n: {:.4f}, -a: {:.4f}'.format(
        losses_n_n_r/cnt, losses_n_a_r/cnt, losses_n_n_w/cnt, losses_n_a_w/cnt))
    print('Test: Adversarial MI: Right samples: -n: {:.4f}, -a: {:.4f}; Wrong samples: -n: {:.4f}, -a: {:.4f}'.format(
        losses_a_n_r / cnt, losses_a_a_r / cnt, losses_a_n_w / cnt, losses_a_a_w / cnt))

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
    local_a = torch.nn.DataParallel(local_a).cuda()
    global_a = torch.nn.DataParallel(global_a).cuda()

    cudnn.benchmark = True

    opt_local_n, schedule_local_n = make_optimizer_and_schedule(local_n, lr=args.lr_mi)
    opt_global_n, schedule_global_n = make_optimizer_and_schedule(global_n, lr=args.lr_mi)
    opt_local_a, schedule_local_a = make_optimizer_and_schedule(local_a, lr=args.lr_mi)
    opt_global_a, schedule_global_a = make_optimizer_and_schedule(global_a, lr=args.lr_mi)

    # Train
    for epoch in range(1, args.epochs + 1):
        loss_n_all = 0
        loss_a_all = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # craft adversarial examples
            adv = craft_adversarial_example_pgd(model=target_model, x_natural=data, y=target, step_size=0.007,
                epsilon=0.5, perturb_steps=20, distance='l_2')

            # Train MI estimator
            loss_n = MI_loss_nat(i=batch_idx, model=target_model, x_natural=data, y=target, x_adv=adv,
                           local_n=local_n, global_n=global_n, epoch=epoch)

            loss_n_all += loss_n

            opt_local_n.zero_grad()
            opt_global_n.zero_grad()
            loss_n.backward()
            opt_local_n.step()
            opt_global_n.step()

            loss_a = MI_loss_adv(i=batch_idx, model=target_model, x_natural=data, y=target, x_adv=adv,
                                 local_n=local_a, global_n=global_a, epoch=epoch)
            loss_a_all += loss_a

            opt_local_a.zero_grad()
            opt_global_a.zero_grad()
            loss_a.backward()
            opt_local_a.step()
            opt_global_a.step()


        schedule_local_n.step()
        schedule_global_n.step()
        schedule_local_a.step()
        schedule_global_a.step()

        loss_n_all = loss_n_all / (batch_idx +1)
        loss_a_all = loss_a_all / (batch_idx + 1)

        # evaluation
        print('================================================================')
        # _ = eval_train(model=target_model, device=device, test_loader=train_loader, local_n=local_n,
        #              global_n=global_n)

        test_accuracy = eval_test(model=target_model, device=device, test_loader=test_loader, local_n=local_n,
                                  global_n=global_n, local_a=local_a,
                                  global_a=global_a)

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
            torch.save(local_a.module.state_dict(),
                       os.path.join(args.model_dir, 'local_a_model.pth'))
            torch.save(global_a.module.state_dict(),
                       os.path.join(args.model_dir, 'global_a_model.pth'))
            print('save the model')

        print('================================================================')


if __name__ == '__main__':
    main()
