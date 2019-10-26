# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os
from models.model_helper import weights_init
import importlib
from layers.functions.prior_layer import PriorLayer
from dcn.modules.deform_conv import DeformConv, ModulatedDeformConv

def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'models.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        print('Failed to find function: %s', func_name)
        raise

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

class Basic2Conv(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Basic2Conv, self).__init__()
        self.branch1 = BasicConv(in_planes, out_planes, kernel_size=1)
        self.branch2 = BasicConv(out_planes, out_planes, kernel_size=1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x1)

        return x2

def add_dcn_dilas():

    planes = [512,1024,512,256]
    deformable_groups = 1
    conv_layers = []
    for i in range(4):
        conv_layers += [DeformConv(
            planes[i],
            256,
            kernel_size=3,
            stride=1,
            padding=5-i,
            dilation=5-i,
            deformable_groups=deformable_groups,
            bias=False)]
    return conv_layers

def BN_layers():
    bn_layers =[]
    bn_layers += [nn.BatchNorm2d(256,eps=1e-5, momentum=0.01, affine=True)]
    bn_layers += [nn.BatchNorm2d(256,eps=1e-5, momentum=0.01, affine=True)]
    bn_layers += [nn.BatchNorm2d(256,eps=1e-5, momentum=0.01, affine=True)]
    bn_layers += [nn.BatchNorm2d(256,eps=1e-5, momentum=0.01, affine=True)]

    return bn_layers

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def _init_modules(self):
        self.arm_loc.apply(weights_init)
        self.arm_conf.apply(weights_init)
        if self.cfg.MODEL.REFINE:
            self.odm_loc.apply(weights_init)
            self.odm_conf.apply(weights_init)

            self.loc_offset_conv.apply(weights_init)
            self.dcn_convs.apply(weights_init)
        if self.cfg.MODEL.LOAD_PRETRAINED_WEIGHTS:
            weights = torch.load(self.cfg.MODEL.PRETRAIN_WEIGHTS)
            print("load pretrain model {}".format(
                self.cfg.MODEL.PRETRAIN_WEIGHTS))
            if self.cfg.MODEL.TYPE.split('_')[-1] == 'vgg':
                self.extractor.vgg.load_state_dict(weights)
            else:
                self.extractor.load_state_dict(weights, strict=False)

    def __init__(self, cfg):
        super(SSD, self).__init__()
        self.cfg = cfg
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.prior_layer = PriorLayer(cfg)
        self.priorbox = PriorBox(cfg)
        self.priors = self.priorbox.forward()
        self.extractor = get_func(cfg.MODEL.CONV_BODY)(self.size,
                                                       cfg.TRAIN.CHANNEL_SIZE)
        if cfg.MODEL.REFINE:
            self.odm_channels = size_cfg.ODM_CHANNELS
            self.arm_num_classes = 2
            self.odm_loc = nn.ModuleList()
            self.odm_conf = nn.ModuleList()

            self.loc_offset_conv = nn.ModuleList()
            self.dcn_convs = nn.ModuleList(add_dcn_dilas())
            self.bn_layers = nn.ModuleList(BN_layers())

        self.arm_loc = nn.ModuleList()
        self.arm_conf = nn.ModuleList()
        self.arm_channels = size_cfg.ARM_CHANNELS
        self.num_anchors = size_cfg.NUM_ANCHORS
        self.input_fixed = size_cfg.INPUT_FIXED
        self.arm_loc = nn.ModuleList()
        self.arm_conf = nn.ModuleList()

        for i in range(len(self.arm_channels)):
            if cfg.MODEL.REFINE:
                self.arm_loc += [
                    nn.Conv2d(
                        self.arm_channels[i],
                        self.num_anchors[i] * 4,
                        kernel_size=3,
                        padding=1)
                ]
                self.arm_conf += [
                    nn.Conv2d(
                        self.arm_channels[i],
                        self.num_anchors[i] * self.arm_num_classes,
                        kernel_size=3,
                        padding=1)
                ]

                self.loc_offset_conv +=[BasicConv(self.num_anchors[i] * 2, 18, kernel_size=1)]
                self.odm_loc += [nn.Sequential(Basic2Conv(self.odm_channels[i], 512),
                           nn.Conv2d(512, self.num_anchors[i] * 4, kernel_size=3, padding=1))
                                 ]
                self.odm_conf += [
                    nn.Sequential(Basic2Conv(self.odm_channels[i], 512),
                                  nn.Conv2d(512, self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1))
                                 ]
            else:
                self.arm_loc += [
                    nn.Conv2d(
                        self.arm_channels[i],
                        self.num_anchors[i] * 4,
                        kernel_size=3,
                        padding=1)
                ]
                self.arm_conf += [
                    nn.Conv2d(
                        self.arm_channels[i],
                        self.num_anchors[i] * self.num_classes,
                        kernel_size=3,
                        padding=1)
                ]
        if cfg.TRAIN.TRAIN_ON:
            self._init_modules()

    def forward(self, input):

        arm_loc = list()
        arm_conf = list()
        if self.cfg.MODEL.REFINE:
            odm_loc = list()
            odm_conf = list()
            conf = list()
            odm_xs_n = list()
            arm_loc_list = list()
            arm_xs, odm_xs = self.extractor(input)


            for (x, l, c) in zip(arm_xs, self.arm_loc, self.arm_conf):
                arm_loc_conv = l(x)
                cc = c(x)
                conf.append(cc)
                arm_loc_list.append(torch.cat([arm_loc_conv[:,0::4,:,:], arm_loc_conv[:,1::4,:,:]], 1))
                arm_loc.append(arm_loc_conv.permute(0, 2, 3, 1).contiguous())
                arm_conf.append(cc.permute(0, 2, 3, 1).contiguous())


            for (conf_fea, odm_xs_fea) in zip(conf, odm_xs):
                conf_obj = conf_fea[:, 1::2, :, :]
                conf_max, _ = torch.max(conf_obj, dim=1, keepdim=True)
                conf_attention = conf_max.sigmoid()
                odm_xs_fea_n = odm_xs_fea * conf_attention + odm_xs_fea
                odm_xs_n.append(odm_xs_fea_n)

            offset_0 = self.loc_offset_conv[0](arm_loc_list[0])
            d0 = F.relu(self.bn_layers[0](self.dcn_convs[0](odm_xs_n[0], offset_0)), inplace=True)

            offset_1 = self.loc_offset_conv[1](arm_loc_list[1])
            d1 = F.relu(self.bn_layers[1](self.dcn_convs[1](odm_xs_n[1], offset_1)), inplace=True)

            offset_2 = self.loc_offset_conv[2](arm_loc_list[2])
            d2 = F.relu(self.bn_layers[2](self.dcn_convs[2](odm_xs_n[2], offset_2)), inplace=True)

            offset_3 = self.loc_offset_conv[3](arm_loc_list[3])
            d3 = F.relu(self.bn_layers[3](self.dcn_convs[3](odm_xs_n[3], offset_3)), inplace=True)
            odm_xs_new = [d0,d1,d2,d3]

            for (x, l, c) in zip(odm_xs_new, self.odm_loc, self.odm_conf):
                odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
            odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        else:
            arm_xs = self.extractor(input)
        img_wh = (input.size(3), input.size(2))
        feature_maps_wh = [(t.size(3), t.size(2)) for t in arm_xs]

        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        if self.cfg.MODEL.REFINE:
            output = (arm_loc.view(arm_loc.size(0), -1, 4),
                      arm_conf.view(
                          arm_conf.size(0), -1, self.arm_num_classes),
                      odm_loc.view(odm_loc.size(0), -1, 4),
                      odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                      self.priors if self.input_fixed else self.prior_layer(
                          img_wh, feature_maps_wh))
        else:
            output = (arm_loc.view(arm_loc.size(0), -1, 4),
                      arm_conf.view(arm_conf.size(0), -1, self.num_classes),
                      self.priors if self.input_fixed else self.prior_layer(
                          img_wh, feature_maps_wh))
        return output
