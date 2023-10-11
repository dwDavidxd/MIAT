from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from data import data_dataset# , data_noise_dataset, distilled_dataset
from models.vggnet import VGGNet19
from models.resnet_new import ResNet18
# from models import resnet_transition
# from models import resnet
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

from utils_attack.autoattack import AutoAttack
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SpatialTransformAttack
from utils_attack.FWA import LinfFWA
from utils_attack.ti_dim_gpu import TIDIM_Attack

parser = argparse.ArgumentParser(description='PyTorch Test')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--dataset', type=str, help='fmnist,cifar10,svhn', default='cifar10')

parser.add_argument('--epsilon', default=0.031, help='perturbation')
parser.add_argument('--num-steps', default=10, help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, help='perturb step size')

parser.add_argument('--model-dir', default='./checkpoint/resnet_18/MIAT_mart',
                    help='directory of model for saving checkpoint')
parser.add_argument('--print_freq', type=int, default=1)

parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def craft_adversarial_example_pgd(model, x_natural, y, step_size=0.003,
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


def craft_adversarial_example(model, x_natural, y, step_size=0.003,
                epsilon= 8/255, perturb_steps=10, distance='l_inf'):

    '''
    adversary = AutoAttack(model, norm='Linf', eps=8 / 255, version='standard')
    '''

    '''
    adversary = DDNL2Attack(model, nb_iter=20, gamma=0.05, init_norm=1.0, quantize=True, levels=16, clip_min=0.0,
                            clip_max=1.0, targeted=False, loss_fn=None)
    '''

    '''
    adversary = CarliniWagnerL2Attack(
        model, 10, clip_min=0.0, clip_max=1.0, max_iterations=10, confidence=1, initial_const=1, learning_rate=1e-2,
        binary_search_steps=4, targeted=False)
    '''

    '''
    adversary = LinfFWA(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                      eps=8/255, kernel_size=4, lr=0.007, nb_iter=40, dual_max_iter=15, grad_tol=1e-4,
                    int_tol=1e-4, device="cuda", postprocess=False, verbose=True)
    '''

    '''
    adversary = SpatialTransformAttack(
            model, 10, clip_min=0.0, clip_max=1.0, max_iterations=10, search_steps=5, targeted=False)
    '''

    '''
    adversary = TIDIM_Attack(model,
                       decay_factor=1, prob=0.5,
                       epsilon=8/255, steps=40, step_size=0.01,
                       image_resize=33,
                       random_start=False)
    '''



    adversary = TIDIM_Attack(eps=8/255, steps=40, step_size=0.007, momentum=0.1, prob=0.5, clip_min=0.0, clip_max=1.0,
                 device=torch.device('cuda'), low=32, high=32)


    # x_adv = adversary.run_standard_evaluation(x_natural, y, bs=args.batch_size)

    # x_adv = adversary.perturb(x_natural, y)

    x_adv = adversary.perturb(model, x_natural, y)

    return x_adv


def evaluate_at(i, data, label, model):
    model.eval()  # Change model to 'eval' mode.

    data_adv = craft_adversarial_example(model=model, x_natural=data, y=label, step_size=args.step_size,
                                         epsilon=args.epsilon, perturb_steps=40, distance='l_inf')

    logits = model(data)
    outputs = F.softmax(logits, dim=1)

    _, pred = torch.max(outputs.data, 1)

    correct1 = (pred == label).sum()


    logits = model(data_adv)
    outputs = F.softmax(logits, dim=1)

    _, pred = torch.max(outputs.data, 1)

    correct_adv = (pred == label).sum()

    acc1 = 100 * float(correct1) / data.size(0)

    acc_adv = 100 * float(correct_adv) / data.size(0)

    if (i + 1) % args.print_freq == 0:
        print(
            'Iter [%d/%d] Test classifier: Nat Acc: %.4f; Adv Acc: %.4f'
            % (i + 1, 10000 // data.size(0), acc1, acc_adv))

    return [acc1, acc_adv]


def main(args):
    # settings
    setup_seed(args.seed)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    model_dir = args.model_dir

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # setup data loader
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)


    # classifier = resnet.ResNet18(10)
    classifier = ResNet18(10)
    # classifier = VGGNet19()
    classifier.load_state_dict(torch.load(os.path.join(model_dir, 'target_model.pth')))
    classifier = torch.nn.DataParallel(classifier).to(device)

    cudnn.benchmark = True

    # Adversraial test
    print('------Starting test------')

    total_num = 0.
    test_nat_correct = 0.
    test_adv_correct = 0.


    for i, (data, labels) in enumerate(test_loader):
        data = data.cuda()
        labels = labels.cuda()

        classifier.eval()
        test_acc = evaluate_at(i, data, labels, classifier)

        total_num += 1
        test_nat_correct += test_acc[0]
        test_adv_correct += test_acc[1]

    test_nat_correct = test_nat_correct / total_num
    test_adv_correct = test_adv_correct / total_num

    print('Test Classifer: Nat ACC: %.4f; Adv ACC: %.4f' % (test_nat_correct, test_adv_correct))


if __name__ == '__main__':
    main(args)
