# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from models.model_helper import weights_init

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.single_branch = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=2, dilation=2),
                BasicConv(inter_planes, out_planes, kernel_size=(3, 3), stride=1, padding=(1, 1))
                )

    def forward(self, x):
        out = self.single_branch(x)
        return out

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
            x) * x
        return out


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5, conv6,
        nn.ReLU(inplace=True), conv7,
        nn.ReLU(inplace=True)
    ]
    return layers


extras_cfg = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [
        256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'S',
        256
    ],
}

base = {
    '300': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512
    ],
    '512': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512
    ],
}


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [
                    nn.Conv2d(
                        in_channels,
                        cfg[k + 1],
                        kernel_size=(1, 3)[flag],
                        stride=2,
                        padding=1)
                ]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def add_extras_bn(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [
                    BasicConv(
                        in_channels,
                        cfg[k + 1],
                        kernel_size=(1, 3)[flag],
                        stride=2,
                        padding=1)
                ]
            else:
                layers += [BasicConv(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


class VGG16Extractor(nn.Module):
    def __init__(self, size):
        super(VGG16Extractor, self).__init__()
        self.vgg = nn.ModuleList(vgg(base[str(size)], 3))
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(add_extras(extras_cfg[str(size)], 1024))
        # self.extras_bn = nn.ModuleList(add_extras_bn(extras_cfg[str(size)], 1024))
        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)
        # self.extras_bn.apply(weights_init)
        self.vgg.apply(weights_init)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # for k, v in enumerate(self.extras_bn):
        #     x = v(x)
        #     if k % 2 == 1:
        #         sources.append(x)
        return sources


def SSDVgg(size, channel_size='48'):
    return VGG16Extractor(size)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    with torch.no_grad():
        model3 = VGG16Extractor(300)
        model3.eval()
        x = torch.randn(16, 3, 300, 300)
        model3.cuda()
        model3(x.cuda())
        import time
        st = time.time()
        for i in range(1000):
            model3(x.cuda())
        print(time.time() - st)
