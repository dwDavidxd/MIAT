import torch.nn as nn
import torch.nn.functional as F
import torch

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)


def cfg(depth):
    depth_lst = [11, 13, 16, 19]
    assert (depth in depth_lst), "Error : VGGnet depth should be either 11, 13, 16, 19"
    cf_dict = {
        '11': [
            64, 'mp',
            128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'],
        '13': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'
            ],
        '16': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 'mp',
            512, 512, 512, 'mp',
            512, 512, 512, 'mp'
            ],
        '19': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 256, 'mp',
            512, 512, 512, 512, 'mp',
            512, 512, 512, 512, 'mp'
            ],
    }

    return cf_dict[str(depth)]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class VGG(nn.Module):
    def __init__(self, depth, num_classes, num_classes_T):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg(depth))
        self.linear = nn.Linear(512, num_classes)
        
        self.features_T = self._make_layers_T(cfg(depth))
        self.linear_T = nn.Linear(512, num_classes_T)

    def forward(self, x, out_T=False, out_ori=False, out_all=False):
        if out_T == True:
            out_t = self.features_T(x)
            out_t = out_t.view(out_t.size(0), -1)
            out_t = self.linear_T(out_t)
            out_t = out_t.reshape(out_t.size(0), 10, 10)
            out_t = F.softmax(out_t, dim=2)
            T = out_t.float()           
            return T
        
        elif out_ori == True:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
            
        else:
            out_t = self.features_T(x)
            out_t = out_t.view(out_t.size(0), -1)
            out_t = self.linear_T(out_t)
            out_t = out_t.reshape(out_t.size(0), 10, 10)
            out_t = F.softmax(out_t, dim=2)
            T = out_t.float()

            out = self.features(x)
            out = out.view(out.size(0), -1)
            logits = self.linear(out)  # logit output

            pred_labels = F.softmax(logits, dim=1)

            noisy_post = torch.bmm(pred_labels.unsqueeze(1), T).squeeze(1)  # softmax output

            if out_all == False:
                return noisy_post
            else:
                return noisy_post, logits

    def _make_layers(self, cfg):
        layers = []
        in_planes = 3

        for x in cfg:
            if x == 'mp':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [conv3x3(in_planes, x), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_planes = x

        # After cfg convolution
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
        
    def _make_layers_T(self, cfg):  
        layers = []
        in_planes = 3
        
        for x in cfg:
            if x == 'mp':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [conv3x3(in_planes, x), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_planes = x

        # After cfg convolution
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGGNet16():
    return VGG(16, 10, 100)


def VGGNet19():
    return VGG(19, 10, 100)
