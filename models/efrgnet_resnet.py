# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.model_helper import FpnAdapter, weights_init



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

def add_extras(size, in_channel, batch_norm=False):
    # Extra layers added to resnet for feature scaling
    layers = []
    layers += [nn.Conv2d(in_channel, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)]
    return layers

def add_extras_bn(size, in_channel, batch_norm=False):
    # Extra layers added to resnet for feature scaling
    layers = []
    layers += [BasicConv(in_channel, 256, kernel_size=1, stride=1)]
    layers += [BasicConv(256, 256, kernel_size=3, stride=2, padding=1)]
    return layers


class BasicBlock_small(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(BasicBlock_small, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.single_branch = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=(1, 1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(inter_planes, out_planes, kernel_size=(1, 1))
                )

    def forward(self, x):
        out = F.relu(self.single_branch(x) + x)
        return out

def add_trans():
    layers = []
    layers += [nn.Sequential( BasicBlock_small(256, 256),
                              BasicBlock_small(256, 256),
                              BasicConv(256, 512, kernel_size=1),
                             )]
    layers += [BasicConv(256, 1024, kernel_size=1)]
    layers += [BasicConv(256, 512, kernel_size=1)]
    return layers


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

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
                # BasicConv(inter_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
                # BasicConv(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
                aspp_module(inter_planes, out_planes)
                )

    def forward(self, x):
        out = self.single_branch(x)
        return out


class BasicRFB_a(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicRFB_a, self).__init__()
        self.out_channels = out_planes
        self.relu = nn.ReLU(inplace=False)
        self.single_branch = nn.Sequential(
            BasicConv(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1))
        )

    def forward(self, x):
        x = self.relu(x)
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

class RefineResnet(nn.Module):
    def __init__(self, block, num_blocks, size):
        super(RefineResnet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.Norm1 = BasicConv(512, 256, kernel_size=3, padding=1)
        self.Norm2 = BasicConv(1024, 256, kernel_size=3, padding=1)
        self.Norm3 = BasicConv(512, 256, kernel_size=3, padding=1)
        self.Norm4 = BasicConv(256, 256, kernel_size=3, padding=1)

        self.MSCF = MSCF(3, 512, stride=1)


        self.impn1 = IMPN(kernel_size=(2, 2), stride=2, padding=0)
        self.impn2 = IMPN(kernel_size=(2, 2), stride=2, padding=0)
        self.impn3 = IMPN(kernel_size=(2, 2), stride=2, padding=0)


        self.dsc1 = Downsample(256, 1024, stride=2, padding=(1, 1))
        self.dsc2 = Downsample(256, 512, stride=2, padding=(1, 1))
        self.dsc3 = Downsample(256, 256, stride=2, padding=(1, 1))


        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.inchannel = block.expansion * 512
        self.extras = nn.ModuleList(add_extras_bn(str(size), self.inchannel))
        self.trans = nn.ModuleList(add_trans())
        self. smooth1 = nn.Conv2d(
            self.inchannel, 512, kernel_size=3, stride=1, padding=1)

        self._init_modules()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.trans.apply(weights_init)
        self.smooth1.apply(weights_init)

        self.Norm1.apply(weights_init)
        self.Norm2.apply(weights_init)
        self.Norm3.apply(weights_init)
        self.Norm4.apply(weights_init)

        self.MSCF.apply(weights_init)


        self.impn1.apply(weights_init)
        self.impn2.apply(weights_init)
        self.impn3.apply(weights_init)


        self.dsc1.apply(weights_init)
        self.dsc2.apply(weights_init)
        self.dsc3.apply(weights_init)


    def forward(self, x):
        # pool images
        x_pool1 = self.impn1(x)
        x_pool2 = self.impn2(x_pool1)
        x_pool3 = self.impn3(x_pool2)

        # Bottom-up
        odm_sources = list()
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        d3 = self.Norm1(c3 * self.MSCF(x_pool3))
        c4 = self.layer3(c3)
        d4 = self.Norm2(self.dsc1(d3) * c4)
        c5 = self.layer4(c4)
        x = c5
        c5_ = self.smooth1(c5)
        d5 = self.Norm3(c5_ * self.dsc2(d4))
        # arm_sources = [c3, c4, c5_]
        a3 = self.trans[0](d3)
        a4 = self.trans[1](d4)
        a5 = self.trans[2](d5)
        arm_sources = [a3, a4, a5]
        for k, v in enumerate(self.extras):
            # x = F.relu(v(x), inplace=True)
            x = v(x)
            if k % 2 == 1:
                d6 = self.Norm4(x * self.dsc3(d5))
                arm_sources.append(d6)

        # odm_sources = self.fpn(arm_sources)
        odm_sources = arm_sources
        return arm_sources, odm_sources


def RefineResnet50(size, channel_size='48'):
    return RefineResnet(Bottleneck, [3, 4, 6, 3], size)


def RefineResnet101(size, channel_size='48'):
    return RefineResnet(Bottleneck, [3, 4, 23, 3], size)


def RefineResnet152(size, channel_size='48'):
    return RefineResnet(Bottleneck, [3, 8, 36, 3], size)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = RefineResnet50(size=300)
    print(model)
    with torch.no_grad():
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        model.cuda()
        model(x.cuda())
