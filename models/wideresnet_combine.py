import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, num_classes_T=100, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # 1st conv before any network block
        self.conv1_T = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1_T = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 1st sub-block
        self.sub_block1_T = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2_T = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3_T = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1_T = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc_T = nn.Linear(nChannels[3], num_classes_T)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, out_T=False, out_ori=False, out_all=False):
        if out_T == True:
            out_T = self.conv1_T(x)
            out_T = self.block1_T(out_T)
            out_T = self.block2_T(out_T)
            out_T = self.block3_T(out_T)
            out_T = self.relu(self.bn1_T(out_T))
            out_T = F.avg_pool2d(out_T, 8)
            out_T = out_T.view(-1, self.nChannels)
            out_T = self.fc_T(out_T)

            out_T = out_T.reshape(out_T.size(0), 10, 10)
            out_T = F.softmax(out_T, dim=2)
            T = out_T.float()
            return T

        elif out_ori == True:
            out = self.conv1(x)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)
            return self.fc(out)

        else:
            out_T = self.conv1_T(x)
            out_T = self.block1_T(out_T)
            out_T = self.block2_T(out_T)
            out_T = self.block3_T(out_T)
            out_T = self.relu(self.bn1_T(out_T))
            out_T = F.avg_pool2d(out_T, 8)
            out_T = out_T.view(-1, self.nChannels)
            out_T = self.fc_T(out_T)

            out_T = out_T.reshape(out_T.size(0), 10, 10)
            out_T = F.softmax(out_T, dim=2)
            T = out_T.float()

            out = self.conv1(x)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)
            logits = self.fc(out)

            pred_labels = F.softmax(logits, dim=1)

            noisy_post = torch.bmm(pred_labels.unsqueeze(1), T).squeeze(1)  # softmax output

            if out_all == False:
                return noisy_post
            else:
                return noisy_post, logits


