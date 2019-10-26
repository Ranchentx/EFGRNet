# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from models.model_helper import FpnAdapter,FpnAdapter_bn, WeaveAdapter, weights_init
from mmdet.ops import DeformConv, ModulatedDeformConv


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
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

def add_dilas():
    layers = []
    layers += [BasicConv(512, 256, kernel_size=3, dilation=5, padding=5)]
    layers += [BasicConv(1024, 256, kernel_size=3, dilation=4, padding=4)]
    layers += [BasicConv(256, 256, kernel_size=3, dilation=3, padding=3)]
    layers += [BasicConv(256, 256, kernel_size=3, dilation=2, padding=2)]

    return layers

def add_trans():
    layers = []
    layers += [BasicConv(512, 256, kernel_size=3, padding=1)]
    layers += [BasicConv(1024, 256, kernel_size=3, padding=1)]
    layers += [BasicConv(256, 256, kernel_size=3, padding=1)]
    layers += [BasicConv(256, 256, kernel_size=3, padding=1)]
    return layers

class aspp_module(nn.Module):
    def __init__(self, in_planes, out_planes, r1=2, r2=4):
        super(aspp_module,self).__init__()
        inter_planes = in_planes//4
        self.branch_1 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1),
                                     BasicConv(inter_planes, inter_planes, kernel_size=3, padding=1))
        self.branch_2 = nn.Sequential( BasicConv(in_planes, inter_planes, kernel_size=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, padding=r1, dilation=r1))
        self.branch_3 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1),
                                      BasicConv(inter_planes, inter_planes, kernel_size=3, padding=r2, dilation=r2))
        self.branch_4 = BasicConv(3*inter_planes, out_planes, kernel_size=1, relu=False)
    def forward(self,x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        out = self.branch_4(torch.cat([x1,x2,x3], 1))
        return out

class MSCF(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(MSCF, self).__init__()
        self.out_channels = out_planes
        inter_planes = out_planes // 4
        self.single_branch = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
                BasicConv(inter_planes, inter_planes, kernel_size=1, stride=1),
                aspp_module(inter_planes, out_planes)
                )

    def forward(self, x):
        out = self.single_branch(x)
        return out


class IMPN(nn.Module):

    def __init__(self, kernel_size, stride, padding):
        super(IMPN, self).__init__()
        self.single_branch = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.single_branch(x)
        return out


class IBN(nn.Module):

    def __init__(self, out_planes, bn=True):
        super(IBN, self).__init__()
        self.out_channels = out_planes
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        return x


class BasicRFB_b(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicRFB_b, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.single_branch = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), relu=False)
                )

    def forward(self, x):
        out = self.single_branch(x)
        return out


class BasicRFB_a(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, stride=1):
        super(BasicRFB_a, self).__init__()
        self.out_channels = out_planes
        self.relu = nn.ReLU(inplace=False)
        self.single_branch = nn.Sequential(
            BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        x = self.relu(x)
        out = self.single_branch(x)
        return out

class BasicRFB_c(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BasicRFB_c, self).__init__()
        self.out_channels = out_planes
        self.relu = nn.ReLU(inplace=False)
        self.single_branch = nn.Sequential(
            BasicBlock(in_planes, out_planes)
        )

    def forward(self, x):
        x = self.relu(x)
        out = self.single_branch(x)
        return out

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.single_branch = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=(1, 1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(inter_planes, out_planes, kernel_size=(1, 1))
                )

    def forward(self, x):
        out = self.single_branch(x)
        return out

class Downsample(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, padding=(1, 1)):
        super(Downsample, self).__init__()
        self.out_channels = out_planes
        self.single_branch = nn.Sequential(
            BasicConv(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=padding, relu=False)
        )

    def forward(self, x):
        out = self.single_branch(x)
        return out


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


def add_extras(size):
    layers = []
    layers += [BasicConv(1024, 256, kernel_size=1, stride=1)]
    layers += [BasicConv(256, 256, kernel_size=3, stride=2, padding=1)]
    layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
    layers += [BasicConv(128, 256, kernel_size=3, stride=2, padding=1)]

    return layers




class VGG16Extractor(nn.Module):
    def __init__(self, size, channel_size='48'):
        super(VGG16Extractor, self).__init__()
        self.vgg = nn.ModuleList(vgg(base[str(size)], 3))
        self.extras = nn.ModuleList(add_extras(str(size)))

        # conv_4
        self.Norm1 = BasicRFB_a(512, 512, kernel_size=3, padding=1)
        self.Norm2 = BasicRFB_a(1024, 1024, kernel_size=3, padding=1)
        self.Norm3 = BasicConv(256, 256, kernel_size=3, padding=1)
        self.Norm4 = BasicConv(256, 256, kernel_size=3, padding=1)

        self.icn1 = MSCF(3, 512, stride=1)

        self.impn1 = IMPN(kernel_size=(2, 2), stride=2, padding=0)
        self.impn2 = IMPN(kernel_size=(2, 2), stride=2, padding=0)
        self.impn3 = IMPN(kernel_size=(2, 2), stride=2, padding=0)

        self.dsc1 = Downsample(512, 1024, stride=2, padding=(1, 1))
        self.dsc2 = Downsample(1024, 256, stride=2, padding=(1, 1))
        self.dsc3 = Downsample(256, 256, stride=2, padding=(1, 1))


        self.ibn1 = IBN(512, bn=True)
        self.ibn2 = IBN(1024, bn=True)

        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.Norm1.apply(weights_init)
        self.Norm2.apply(weights_init)
        self.Norm3.apply(weights_init)
        self.Norm4.apply(weights_init)

        self.icn1.apply(weights_init)

        self.impn1.apply(weights_init)
        self.impn2.apply(weights_init)
        self.impn3.apply(weights_init)

        self.dsc1.apply(weights_init)
        self.dsc2.apply(weights_init)
        self.dsc3.apply(weights_init)

        self.ibn1.apply(weights_init)
        self.ibn2.apply(weights_init)


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
        arm_sources = list()
        fbb = list()

        x_pool1 = self.impn1(x)
        x_pool2 = self.impn2(x_pool1)
        x_pool3 = self.impn3(x_pool2)


        for i in range(23):
            x = self.vgg[i](x)
        #38x38
        fbb.append(x)
        att1 = self.icn1(x_pool3)
        mm1 = self.ibn1(x) * att1
        # c2 = self.ibn1(x) * att1
        c2 = self.Norm1(mm1)
        arm_sources.append(c2)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        #19x19
        fbb.append(x)
        att2 = self.dsc1(c2)
        mm2 = self.ibn2(x) * att2
        # c3 = self.dsc1(c2) * att2
        c3 = self.Norm2(mm2)
        arm_sources.append(c3)

        # 10x10
        x = self.extras[0](x)
        x = self.extras[1](x)
        fbb.append(x)
        att3 = self.dsc2(c3)
        mm3 = x * att3
        # c4 = x * att3
        c4 = self.Norm3(mm3)
        arm_sources.append(c4)

        # 5x5
        x = self.extras[2](x)
        x = self.extras[3](x)
        fbb.append(x)
        att4 = self.dsc3(c4)
        mm4 = x * att4
        # c5 = x * att4
        c5 = self.Norm4(mm4)
        arm_sources.append(c5)

        if len(self.extras) > 4:
            x = F.relu(self.extras[4](x), inplace=True)
            x = F.relu(self.extras[5](x), inplace=True)
            c6 = x
            arm_sources.append(c6)

        att = [att1, att2, att3, att4]
        mm = [mm1,mm2,mm3,mm4]
        odm_sources = arm_sources
        # return arm_sources
        # return arm_sources, odm_sources, fbb, att, mm
        # return arm_sources, fbb, att, mm
        return arm_sources, odm_sources


def refine_vgg(size, channel_size='48'):
    return VGG16Extractor(size)
