"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

from .method import AcolBase
from .method import ADL
from .method import spg
from .method.util import normalize_tensor, get_attention
from .util import remove_layer
from .util import replace_layer
from .util import initialize_weights
import numpy as np
import torch.nn.functional as F

__all__ = ['resnet34', 'resnet50']

model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

_ADL_POSITION = [[], [], [], [0], [0, 2]]


#----------------------add---------------------------
# def erase_feature_maps(atten_map_normed, feature_maps, threshold=0.8):
#     # atten_map_normed = torch.unsqueeze(atten_map_normed, dim=1)
#     # atten_map_normed = self.up_resize(atten_map_normed)
#     # if len(atten_map_normed.size())>3:
#     #     atten_map_normed = torch.squeeze(atten_map_normed)
#     atten_shape = atten_map_normed.size()
#     pos = torch.ge(atten_map_normed, threshold)
#     mask = torch.ones(atten_shape).cuda()
#     mask[pos.data] = 0.0
#     # mask = torch.unsqueeze(mask, dim=1)
#     #erase
#     erased_feature_maps = feature_maps * mask

#     return erased_feature_maps


def erase_attmap(att_map, threshold=0.8):
    att_size = att_map.size()
    pos = torch.ge(att_map, threshold)
    mask = torch.ones(att_size).cuda()
    mask[pos.data] = 0.0
    erased_attmap = att_map * mask
    return erased_attmap
#---------------------------------------------------


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetBCAM(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, num_head=100, **kwargs):
        super(ResNetBCAM, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.label = None
        self.drop_threshold = kwargs['acol_drop_threshold']

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #--------------add----------------------------------------
        # self.fc51 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        # self.bn51 = nn.BatchNorm2d(128)
        # self.fc52 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        # self.fc41 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        # self.bn41 = nn.BatchNorm2d(128)
        # self.fc42 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        # self.fc31 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        # self.bn31 = nn.BatchNorm2d(128)
        # self.fc32 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        # self.linear5 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        # self.bn5 = nn.BatchNorm2d(128)
        # # self.linear4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        # # self.bn4 = nn.BatchNorm2d(128)
        # # self.linear3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        # # self.bn3 = nn.BatchNorm2d(128)

        # self.aggregator_A = nn.Conv2d(128, num_head, 1, bias=False)
        # self.aggregator_B = nn.Conv2d(128, num_head, 1, bias=False)

        # self.classifier_A = nn.Conv2d(128, num_classes, 1, 1, padding=0)
        # self.classifier_B = nn.Conv2d(128, num_classes, 1, 1, padding=0)
        #---------------------------------------------------------
        
        self.aggregator_A = nn.Conv2d(512 * block.expansion, num_head, 1, bias=False)
        self.aggregator_B = nn.Conv2d(512 * block.expansion, num_head, 1, bias=False)

        self.classifier_A = nn.Conv2d(512 * block.expansion, num_classes, 1, 1, padding=0)
        self.classifier_B = nn.Conv2d(512 * block.expansion, num_classes, 1, 1, padding=0)
        
        #---------------add-----------------------------
        # # self.fc = torch.nn.Linear(512 * block.expansion, num_classes)
        # self.classifier_A_erase = nn.Conv2d(512 * block.expansion, num_classes, 1, 1, padding=0)
        # self.classifier_B_erase = nn.Conv2d(512 * block.expansion, num_classes, 1, 1, padding=0)
        #-----------------------------------------------

        initialize_weights(self.modules(), init_mode='he')
        initialize_weights(self.aggregator_A.modules(), init_mode='contant')
        initialize_weights(self.aggregator_B.modules(), init_mode='contant')


    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature = x.clone()

        #----------------------add---------------------------
        # out2 = self.layer1(x)
        # out3 = self.layer2(out2)
        # out4 = self.layer3(out3)
        # out5 = self.layer4(out4)
        # out5a = F.relu(self.bn51(self.fc51(out5.mean(dim=(2, 3), keepdim=True))), inplace=True)
        # out4a = F.relu(self.bn41(self.fc41(out4.mean(dim=(2, 3), keepdim=True))), inplace=True)
        # out3a = F.relu(self.bn31(self.fc31(out3.mean(dim=(2, 3), keepdim=True))), inplace=True)
        # vector = out5a * out4a * out3a
        # out5 = torch.sigmoid(self.fc52(vector)) * out5
        # # out4 = torch.sigmoid(self.fc42(vector)) * out4
        # # out3 = torch.sigmoid(self.fc32(vector)) * out3

        # out5 = F.relu(self.bn5(self.linear5(out5)), inplace=True)
        # # out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        # feature = out5.clone()
        #----------------------------------------------------
        
        att_map_A = self.aggregator_A(feature)
        #----------------------add----------------------
        # att_map_A_erase = erase_attmap(att_map_A)
        #---------------------------------------------
        b, c, h, w = att_map_A.shape
        att_map_A = torch.softmax(att_map_A.view(b, c, h*w), dim=-1)

        att_map_B = self.aggregator_B(feature)
        #----------------------add----------------------
        # att_map_B_erase = erase_attmap(att_map_B)
        #---------------------------------------------
        att_map_B = torch.softmax(att_map_B.view(b, c, h*w), dim=-1)

        f_pixel = feature.view(b, -1, h*w)
        f_image_fore = torch.bmm(att_map_A, f_pixel.permute(0, 2, 1)).mean(dim=1).view(b, -1, 1, 1)
        f_image_back = torch.bmm(att_map_B, f_pixel.permute(0, 2, 1)).mean(dim=1).view(b, -1, 1, 1)
        
        label_fore = self.classifier_A(f_image_fore)
        label_fore = label_fore.view(label_fore.size(0), -1)

        label_back = self.classifier_B(torch.bmm(att_map_B, f_pixel.detach().clone().permute(0, 2, 1)).mean(dim=1).view(b, -1, 1, 1))
        label_back = label_back.view(label_back.size(0), -1)
        
        label_fore_rev = self.classifier_B(torch.bmm(att_map_A, f_pixel.detach().clone().permute(0, 2, 1)).mean(dim=1).view(b, -1, 1, 1))
        label_fore_rev = label_fore_rev.view(label_fore.size(0), -1)

        label_back_rev = self.classifier_A(f_image_back)
        label_back_rev = label_back_rev.view(label_fore.size(0), -1)
        
        ############################# add ######################################
        # att_map_A_erase = torch.softmax(att_map_A_erase.view(b, c, h*w), dim=-1)
        # att_map_B_erase = torch.softmax(att_map_B_erase.view(b, c, h*w), dim=-1)

        # f_image_fore_erase = torch.bmm(att_map_A_erase, f_pixel.permute(0, 2, 1)).mean(dim=1).view(b, -1, 1, 1)
        # f_image_back_erase = torch.bmm(att_map_B_erase, f_pixel.permute(0, 2, 1)).mean(dim=1).view(b, -1, 1, 1)
        
        # label_fore_erase = self.classifier_A_erase(f_image_fore_erase)
        # label_fore_erase = label_fore_erase.view(label_fore_erase.size(0), -1)

        # label_back_erase = self.classifier_B_erase(torch.bmm(att_map_B_erase, f_pixel.detach().clone().permute(0, 2, 1)).mean(dim=1).view(b, -1, 1, 1))
        # label_back_erase = label_back_erase.view(label_back_erase.size(0), -1)
        
        # label_fore_rev_erase = self.classifier_B_erase(torch.bmm(att_map_A_erase, f_pixel.detach().clone().permute(0, 2, 1)).mean(dim=1).view(b, -1, 1, 1))
        # label_fore_rev_erase = label_fore_rev_erase.view(label_fore_erase.size(0), -1)

        # label_back_rev_erase = self.classifier_A_erase(f_image_back_erase)
        # label_back_rev_erase = label_back_rev_erase.view(label_fore_erase.size(0), -1)
        # # if labels is not None:
        # #     feat_map_A = self.classifier_A(feature)
        # #     atten_maps = get_attention(feat_map_A, labels)
        # #     feat_erase = erase_feature_maps(atten_maps, feature)
        # #     erase_logit = self.avgpool(feat_erase)
        # #     erase_logit = erase_logit.reshape(erase_logit.size(0), -1)
        # #     erase_logits = self.fc(erase_logit)
        # # else:
        # #     erase_logits = None
        ##############################################################################

        f_pixel = f_pixel.view(b, -1, h, w)
        att_map_A = torch.sum(att_map_A, dim=1).view(b, 1, h, w)
        att_map_B = torch.sum(att_map_B, dim=1).view(b, 1, h, w)

        if return_cam:
            
            feat_map_A = self.classifier_A(feature)
            feat_map_B = self.classifier_B(feature)

            normalized_a = normalize_tensor(
                feat_map_A.detach().clone())
            normalized_b = normalize_tensor(
                feat_map_B.detach().clone())

            cams = normalized_a[range(batch_size), labels]
            cams_back = normalized_b[range(batch_size), labels]

            return label_fore, cams, cams_back, att_map_A.view(b, 1, h, w), att_map_B.view(b, 1, h, w), feat_map_A, feat_map_B
        
        return {'logits': label_fore, 'logits_back':label_back, 'logits_rev':label_fore_rev, 'logits_back_rev':label_back_rev, 'fore_weight':self.classifier_A.weight, 'back_weight':self.classifier_B.weight}
        # ############### add ##################################
        # return {'logits': label_fore, 'logits_back':label_back, 'logits_rev':label_fore_rev, 'logits_back_rev':label_back_rev, \
        #     'logits_erase': label_fore_erase, 'logits_back_erase':label_back_erase, 'logits_rev_erase':label_fore_rev_erase, 'logits_back_rev_erase':label_back_rev_erase, \
        #     'fore_weight':self.classifier_A.weight, 'back_weight':self.classifier_B.weight}
        ##########################################################

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

class ResNetCam(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetCam, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return logits, cams
        return {'logits': logits}

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers


class ResNetAcol(AcolBase):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetAcol, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.label = None
        self.drop_threshold = kwargs['acol_drop_threshold']

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.classifier_A = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, 1, 1, padding=0),
        )
        self.classifier_B = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, 1, 1, padding=0),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)

        logits_dict = self._acol_logits(feature=feature, labels=labels,
                                        drop_threshold=self.drop_threshold)

        if return_cam:
            normalized_a = normalize_tensor(
                logits_dict['feat_map_a'].detach().clone())
            normalized_b = normalize_tensor(
                logits_dict['feat_map_b'].detach().clone())
            feature_map = torch.max(normalized_a, normalized_b)
            cams = feature_map[range(batch_size), labels]
            return logits_dict['logits'], cams

        return logits_dict

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers


class ResNetSpg(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetSpg, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block=block, planes=64,
                                       blocks=layers[0],
                                       stride=1, split=False)
        self.layer2 = self._make_layer(block=block, planes=128,
                                       blocks=layers[1],
                                       stride=2, split=False)
        self.SPG_A1, self.SPG_A2 = self._make_layer(block=block, planes=256,
                                                    blocks=layers[2],
                                                    stride=stride_l3,
                                                    split=True)
        self.layer4 = self._make_layer(block=block, planes=512,
                                       blocks=layers[3],
                                       stride=1, split=False)
        self.SPG_A4 = nn.Conv2d(512 * block.expansion, num_classes,
                                kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

        self.SPG_C = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        initialize_weights(self.modules(), init_mode='xavier')

    def _make_layer(self, block, planes, blocks, stride, split=None):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        first_layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        other_layers = []
        for _ in range(1, blocks):
            other_layers.append(block(self.inplanes, planes))

        if split:
            return nn.Sequential(*first_layers), nn.Sequential(*other_layers)
        else:
            return nn.Sequential(*(first_layers + other_layers))

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.SPG_A1(x)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.SPG_A2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.layer4(x)
        feat_map = self.SPG_A4(x)

        logits_c = self.SPG_C(x)

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention, fused_attention = spg.compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)

        if return_cam:
            feature_map = feat_map.clone().detach()
            cams = feature_map[range(batch_size), labels]
            return logits, cams
        return {'attention': attention, 'fused_attention': fused_attention,
                'logits': logits, 'logits_b1': logits_b1,
                'logits_b2': logits_b2, 'logits_c': logits_c}


class ResNetAdl(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetAdl, self).__init__()

        self.stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.adl_drop_rate = kwargs['adl_drop_rate']
        self.adl_threshold = kwargs['adl_drop_threshold']

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stride=1,
                                       split=_ADL_POSITION[1])
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=2,
                                       split=_ADL_POSITION[2])
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=self.stride_l3,
                                       split=_ADL_POSITION[3])
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=1,
                                       split=_ADL_POSITION[4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return logits, cams

        return {'logits': logits}

    def _make_layer(self, block, planes, blocks, stride, split=None):
        layers = self._layer(block, planes, blocks, stride)
        for pos in reversed(split):
            layers.insert(pos + 1, ADL(self.adl_drop_rate, self.adl_threshold))
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers


def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )


def align_layer(state_dict):
    keys = [key for key in sorted(state_dict.keys())]
    for key in reversed(keys):
        move = 0
        if 'layer' not in key:
            continue
        key_sp = key.split('.')
        layer_idx = int(key_sp[0][-1])
        block_idx = key_sp[1]
        if not _ADL_POSITION[layer_idx]:
            continue

        for pos in reversed(_ADL_POSITION[layer_idx]):
            if pos < int(block_idx):
                move += 1

        key_sp[1] = str(int(block_idx) + move)
        new_key = '.'.join(key_sp)
        state_dict[new_key] = state_dict.pop(key)
    return state_dict


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'layer3.0.', 'SPG_A1.0.')
    state_dict = replace_layer(state_dict, 'layer3.1.', 'SPG_A2.0.')
    state_dict = replace_layer(state_dict, 'layer3.2.', 'SPG_A2.1.')
    state_dict = replace_layer(state_dict, 'layer3.3.', 'SPG_A2.2.')
    state_dict = replace_layer(state_dict, 'layer3.4.', 'SPG_A2.3.')
    state_dict = replace_layer(state_dict, 'layer3.5.', 'SPG_A2.4.')
    return state_dict


def load_pretrained_model(model, wsol_method, path=None, **kwargs):
    strict_rule = True

    if path:
        # state_dict = torch.load(os.path.join(path, 'resnet50.pth'))
        state_dict = torch.load(path)
    else:
        state_dict = load_url(model_urls['resnet50'], progress=True)

    if wsol_method == 'adl':
        state_dict = align_layer(state_dict)
    elif wsol_method == 'spg':
        state_dict = batch_replace_layer(state_dict)

    if kwargs['dataset_name'] != 'ILSVRC' or wsol_method in ('acol', 'spg'):
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False
    elif kwargs['dataset_name'] == 'ILSVRC' and wsol_method in ('bcam'):
        #state_dict = replace_layer(state_dict, 'fc', 'classifier_A')
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def load_pretrained_res34(model, wsol_method, path=None, **kwargs):
    strict_rule = True

    if path:
        state_dict = torch.load(path)
    else:
        state_dict = load_url(model_urls['resnet34'], progress=True)

    if wsol_method == 'adl':
        state_dict = align_layer(state_dict)
    elif wsol_method == 'spg':
        state_dict = batch_replace_layer(state_dict)

    if kwargs['dataset_name'] != 'ILSVRC' or wsol_method in ('acol', 'spg'):
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False
    elif kwargs['dataset_name'] == 'ILSVRC' and wsol_method in ('bcam'):
        #state_dict = replace_layer(state_dict, 'fc', 'classifier_A')
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def resnet50(architecture_type, pretrained=False, pretrained_path=None,
             **kwargs):
    model = {'cam': ResNetCam,
             'acol': ResNetAcol,
             'spg': ResNetSpg,
             'bcam': ResNetBCAM,
             'adl': ResNetAdl}[architecture_type](Bottleneck, [3, 4, 6, 3],
                                                  **kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path, **kwargs)
    return model


def resnet34(architecture_type, pretrained=False, pretrained_path=None,
             **kwargs):
    model = {'cam': ResNetCam,
             'acol': ResNetAcol,
             'spg': ResNetSpg,
             'bcam': ResNetBCAM,
             'adl': ResNetAdl}[architecture_type](BasicBlock, [3, 4, 6, 3],
                                                  **kwargs)
    if pretrained:
        model = load_pretrained_res34(model, architecture_type,
                                      path=pretrained_path, **kwargs)
    return model