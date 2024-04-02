import copy
import numpy as np
from torch.autograd import Variable
import torch
import scipy.stats as st
from scipy import ndimage
import random
import torch.nn.functional as F
import torch.nn as nn

import pdb


def input_diversity(image, prob, low=200, high=224):
    if random.random() > prob:
        return image
    rnd = random.randint(low, high)
    rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
    h_remain = high - rnd
    w_remain = high - rnd
    pad_top = random.randint(0, h_remain)
    pad_bottom = h_remain - pad_top
    pad_left = random.randint(0, h_remain)
    pad_right = w_remain - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    return padded


def gkern(kernlen=21, nsig=3):
    # get 2d Gaussian kernel array
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    # get the normal gaussian distribution pdf on x
    kernel1d = st.norm.pdf(x)
    kernel_raw = np.outer(kernel1d, kernel1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def conv2d_same_padding(inputs, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = inputs.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation + 1
    out_rows = (input_rows + stride - 1) // stride
    padding_rows = max(0, (out_rows - 1) * stride +
                       (filter_rows - 1) * dilation + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride +
                       (filter_rows - 1) * dilation + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        inputs = F.pad(inputs, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(inputs, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


class Attacker:
    def __init__(self, eps: float = 8.0 / 255, clip_min: float = 0.0, clip_max: float = 1.0,
                 device: torch.device = torch.device('cpu')):
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.eps = eps
        self.device = device

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass


class TIDIM_Attack(Attacker):
    def __init__(self, eps, steps, step_size, momentum, prob=0.5, clip_min=0.0, clip_max=1.0,
                 device=torch.device('cpu'), low=224,
                 high=240):
        super(TIDIM_Attack, self).__init__(eps=eps, clip_min=clip_min, clip_max=clip_max, device=device)
        self.steps = steps
        self.step_size = step_size
        self.momentum = momentum
        self.prob = prob
        self.low = low
        self.high = high
        self.loss_func = F.cross_entropy

    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        model.eval()
        # nx = torch.unsqueeze(x, 0).to(self.device)
        # ny = torch.unsqueeze(y, 0).to(self.device)
        nx = x.to(self.device)
        ny = y.to(self.device)
        nx.requires_grad_(True)

        eta = torch.zeros(nx.shape).to(self.device)
        adv_t = nx + eta

        g = 0

        # get the conv pre-defined kernel
        kernel = gkern(15, 3).astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)  # shape: [3, 1, 15, 15]
        conv_weight = torch.from_numpy(stack_kernel).to(self.device)  # kernel weight for depth_wise convolution

        for i in range(self.steps):
            adv_diversity = input_diversity(adv_t, prob=self.prob, low=self.low, high=self.high)
            adv_normalize = adv_diversity
            out = model(adv_normalize)
            loss = self.loss_func(out, ny)
            loss.backward()

            gradient = nx.grad.data
            # (padding = SAME) in tensorflow
            ti_gradient = conv2d_same_padding(gradient, weight=conv_weight, stride=1, padding=1, groups=3)
            ti_gradient = ti_gradient / torch.mean(torch.abs(ti_gradient), [1, 2, 3], keepdim=True)
            g = self.momentum * g + ti_gradient
            eta += self.step_size * torch.sign(g)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()
            adv_t = nx + eta
            adv_t.clamp_(self.clip_min, self.clip_max)

        return adv_t.squeeze(0).detach()
