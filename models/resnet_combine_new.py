#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:46:26 2020

@author: pc-3
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):

        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
         
    def forward(self, x):
        x = x.mm(self.w)
        return x 


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_blocks_T, num_classes, num_classes_T):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.in_planes_T = 64
        self.T_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.T_bn1 = nn.BatchNorm2d(64)
        self.T_layer1 = self._make_layer_T(block, 64, num_blocks_T[0], stride=1)
        self.T_layer2 = self._make_layer_T(block, 128, num_blocks_T[1], stride=2)
        self.T_layer3 = self._make_layer_T(block, 256, num_blocks_T[2], stride=2)
        self.T_layer4 = self._make_layer_T(block, 512, num_blocks_T[3], stride=2)
        self.T_bayes_linear = nn.Linear(512 * block.expansion, num_classes_T)
        self.T_avgpool = nn.AdaptiveAvgPool2d((1, 1))
           
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_T(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_T, planes, stride))
            self.in_planes_T = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_T=False, out_ori=False, out_all=False):
        if out_T == True:
            out_T = F.relu(self.T_bn1(self.T_conv1(x)))
            out_T = self.T_layer1(out_T)
            out_T = self.T_layer2(out_T)
            out_T = self.T_layer3(out_T)
            out_T = self.T_layer4(out_T)
            out_T = self.T_avgpool(out_T)
            out_1_T = out_T.view(out_T.size(0), -1)
            out_2_T = self.T_bayes_linear(out_1_T)
            out_2_T = out_2_T.reshape(out_2_T.size(0), 10, 10)
            out_2_T = F.softmax(out_2_T, dim=2)
            T = out_2_T.float()
            return T

        elif out_ori == True:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out_1 = out.view(out.size(0), -1)
            logits = self.linear(out_1) # logit output
            return logits

        else:
            out_T = F.relu(self.T_bn1(self.T_conv1(x)))
            out_T = self.T_layer1(out_T)
            out_T = self.T_layer2(out_T)
            out_T = self.T_layer3(out_T)
            out_T = self.T_layer4(out_T)
            out_T = self.T_avgpool(out_T)
            out_1_T = out_T.view(out_T.size(0), -1)
            out_2_T = self.T_bayes_linear(out_1_T)
            out_2_T = out_2_T.reshape(out_2_T.size(0), 10, 10)
            out_2_T = F.softmax(out_2_T, dim=2)
            T = out_2_T.float()

            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out_1 = out.view(out.size(0), -1)
            logits = self.linear(out_1)  # logit output

            pred_labels = F.softmax(logits, dim=1)

            noisy_post = torch.bmm(pred_labels.unsqueeze(1), T).squeeze(1)  # softmax output

            if out_all == False:
                return noisy_post
            else:
                return noisy_post, logits


def ResNet18(num_classes, num_classes_T):
    return ResNet(BasicBlock, [2,2,2,2], [2,2,2,2], num_classes, num_classes_T)
    
def ResNet34(num_classes, num_classes_T):
    return ResNet(BasicBlock, [3,4,6,3], [3,4,6,3], num_classes, num_classes_T)

