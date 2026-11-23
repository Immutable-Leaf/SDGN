import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18
import numpy as np
import math


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm.clamp_min(1e-12))
        return out



def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class att_resnet(nn.Module):
    def __init__(self, class_num, arch='resnet50', pretrained=True):
        super(att_resnet, self).__init__()

        model_base = resnet50(pretrained=pretrained,
                              last_conv_stride=1, last_conv_dilation=1)

        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.SA = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        f = self.base.layer4(x)
        x = torch.mul(x, self.sigmoid(torch.mean(f, dim=1, keepdim=True)))
        f = self.base.avgpool(f).flatten(1)
        out, feat = self.classifier(f)
        return x, out, feat


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50', pretrained=True):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=pretrained,
                              last_conv_stride=1, last_conv_dilation=1)

        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        xL1 = self.base.layer1(x)
        xL2 = self.base.layer2(xL1)
        xL3 = self.base.layer3(xL2)
        return xL1, xL2, xL3


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        x = self.classifier(x)
        return x, f


class classifier(nn.Module):
    def __init__(self, num_part, class_num):
        super(classifier, self).__init__()
        input_dim = 1024
        self.part = int(num_part)
        self.l2norm = Normalize(2)
        for i in range(self.part):
            name = 'classifier_' + str(i)
            setattr(self, name, ClassBlock(input_dim, class_num))

    def forward(self, x, feat_all, out_all):
        start_point = len(feat_all)
        for i in range(self.part):
            name = 'classifier_' + str(i)
            cls_part = getattr(self, name)
            out_all[i + start_point], feat_all[i + start_point] = cls_part(x[:, :, i].flatten(1))
            feat_all[i + start_point] = self.l2norm(feat_all[i + start_point])

        return feat_all, out_all


class SDMA(nn.Module):

    def __init__(self, shallow_channels, deep_channels, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        group_channels = shallow_channels // num_experts
        output_channels = deep_channels // num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(group_channels, group_channels, 3, padding=1),
                nn.BatchNorm2d(group_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(group_channels, output_channels, 1),
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(shallow_channels, num_experts, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x, target_size):
        x_group = x.chunk(self.num_experts, dim=1)
        gate = self.gate(x)
        weight = []
        for i, expert in enumerate(self.experts):
            x = expert(x_group[i])
            x = F.adaptive_avg_pool2d(x, target_size)
            w = torch.sigmoid(x)
            w = self.num_experts * gate[:, i:i + 1] * w
            weight.append(w)
        weight = torch.cat(weight, dim=1)
        return weight


class L1ToL2Adaptor(SDMA):
    def __init__(self, num_experts=4):
        super().__init__(256, 512, num_experts)

    def forward(self, xL1, target_size):
        return super().forward(xL1, target_size)


class L2ToL3Adaptor(SDMA):
    def __init__(self, num_experts=4):
        super().__init__(512, 1024, num_experts)

    def forward(self, xL2, target_size):
        return super().forward(xL2, target_size)


class embed_net(nn.Module):
    def __init__(self, class_num, part, arch='resnet50',
                 num_experts=4, pretrained=True):
        super(embed_net, self).__init__()

        self.part = int(part)
        self.alpha1 = 0.3
        self.alpha2 = 0.3
        self.base_resnet = base_resnet(arch=arch, pretrained=pretrained)
        self.att_v = att_resnet(class_num, arch=arch, pretrained=pretrained)
        self.att_n = att_resnet(class_num, arch=arch, pretrained=pretrained)
        self.classifier = classifier(self.part, class_num)

        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.adaptor12_v = L1ToL2Adaptor(num_experts)
        self.adaptor12_n = L1ToL2Adaptor(num_experts)
        self.adaptor23_v = L2ToL3Adaptor(num_experts)
        self.adaptor23_n = L2ToL3Adaptor(num_experts)



    def forward(self, x1, x2, modal=0):
        if modal == 0:
            batch_v, batch_n = x1.size(0), x2.size(0)
            x = torch.cat((x1, x2), 0)
            xL1, xL2, xL3 = self.base_resnet(x)


            xL1_v, xL1_n = xL1.split((batch_v, batch_n), dim=0)
            xL2_v, xL2_n = xL2.split((batch_v, batch_n), dim=0)
            w_v12 = self.adaptor12_v(xL1_v, xL2_v.shape[-2:])
            w_n12 = self.adaptor12_n(xL1_n, xL2_n.shape[-2:])
            xL2_v = xL2_v * (1 + self.alpha1 * w_v12)
            xL2_n = xL2_n * (1 + self.alpha1 * w_n12)


            xL3_v, xL3_n = xL3.split((batch_v, batch_n), dim=0)
            w_v23 = self.adaptor23_v(xL2_v, xL3_v.shape[-2:])
            w_n23 = self.adaptor23_n(xL2_n, xL3_n.shape[-2:])
            xL3_v = xL3_v * (1 + self.alpha2 * w_v23)
            xL3_n = xL3_n * (1 + self.alpha2 * w_n23)


            x1_att, out_v, feat_v = self.att_v(xL3_v)
            x2_att, out_n, feat_n = self.att_n(xL3_n)

            x = torch.cat((x1_att, x2_att), 0)
            feat_globe = torch.cat((feat_v, feat_n), 0)
            out_globe = torch.cat((out_v, out_n), 0)

        elif modal == 1:
            xL1, xL2, xL3 = self.base_resnet(x1)
            w_v12 = self.adaptor12_v(xL1, xL2.shape[-2:])
            xL2 = xL2 * (1 + self.alpha1 * w_v12)
            w_v23 = self.adaptor23_v(xL2, xL3.shape[-2:])
            xL3 = xL3 * (1 + self.alpha2 * w_v23)
            x, _, _ = self.att_v(xL3)

        elif modal == 2:
            xL1, xL2, xL3 = self.base_resnet(x2)
            w_n12 = self.adaptor12_n(xL1, xL2.shape[-2:])
            xL2 = xL2 * (1 + self.alpha1 * w_n12)
            w_n23 = self.adaptor23_n(xL2, xL3.shape[-2:])
            xL3 = xL3 * (1 + self.alpha2 * w_n23)
            x, _, _ = self.att_n(xL3)


        x = self.avgpool(x)
        feat = {}
        out = {}
        feat, out = self.classifier(x, feat, out)
        if self.training:
            return (
                feat, out, feat_globe, out_globe,
            )
        else:
            featf = torch.cat([feat[i] for i in range(self.part)], 1)
            return featf
