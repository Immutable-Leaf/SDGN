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
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
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
    def __init__(self, class_num, arch='resnet50'):
        super(att_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.SA = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        f = self.base.layer4(x)
        x = torch.mul(x, self.sigmoid(torch.mean(f, dim=1, keepdim=True)))
        f = torch.squeeze(self.base.avgpool(f))
        out, feat = self.classifier(f)
        return x, out, feat


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        xL1 = self.base.layer1(x)
        xL2 = self.base.layer2(xL1)      # (B,512,48,24)
        xL3 = self.base.layer3(xL2)      # (B,1024,24,12)
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
            out_all[i + start_point], feat_all[i + start_point] = cls_part(torch.squeeze(x[:, :, i]))
            feat_all[i + start_point] = self.l2norm(feat_all[i + start_point])

        return feat_all, out_all


class L1ToL2Adaptor(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1)
        )

    def forward(self, xL1, target_size):
        # xL1: (B,256,96,48)
        x = self.conv(xL1)                         # (B,512,96,48)
        x = F.adaptive_avg_pool2d(x, target_size)  # (B,512,48,24)
        weight = torch.sigmoid(x)
        return weight



class L2ToL3Adaptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 1)   
        )

    def forward(self, xL2, target_size):
                                                   # xL2: (B,512,48,24)
        x = self.conv(xL2)                         # (B,1024,48,24)
        x = F.adaptive_avg_pool2d(x, target_size)  # (B,1024,24,12)
        weight = torch.sigmoid(x)     
        return weight

class embed_net(nn.Module):
    def __init__(self, class_num, part, arch='resnet50'):
        super(embed_net, self).__init__()

        self.part = int(part)
        self.base_resnet = base_resnet(arch=arch)
        self.att_v = att_resnet(class_num)
        self.att_n = att_resnet(class_num)
        self.classifier = classifier(self.part, class_num)

        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.adaptor12_v = L1ToL2Adaptor()
        self.adaptor12_n = L1ToL2Adaptor()
        self.adaptor23_v = L2ToL3Adaptor()
        self.adaptor23_n = L2ToL3Adaptor()



    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x = torch.cat((x1, x2), 0)                    # (2B,3,384,192)
            xL1, xL2, xL3 = self.base_resnet(x)           # xL2: (2B,512,48,24), xL3: (2B,1024,24,12)

            # L1->L2
            xL1_v, xL1_n = torch.chunk(xL1, 2, 0)
            xL2_v, xL2_n = torch.chunk(xL2, 2, 0)
            w_v12 = self.adaptor12_v(xL1_v, xL2_v.shape[-2:])
            w_n12 = self.adaptor12_n(xL1_n, xL2_n.shape[-2:])
            xL2_v = xL2_v * (1 + 0.3 * w_v12)
            xL2_n = xL2_n * (1 + 0.3 * w_n12)

            # L2->L3
            xL3_v, xL3_n = torch.chunk(xL3, 2, 0)
            w_v23 = self.adaptor23_v(xL2_v, xL3_v.shape[-2:])
            w_n23 = self.adaptor23_n(xL2_n, xL3_n.shape[-2:])
            xL3_v = xL3_v * (1 + 0.3 * w_v23)
            xL3_n = xL3_n * (1 + 0.3 * w_n23)

            # L4
            x1_att, out_v, feat_v = self.att_v(xL3_v)
            x2_att, out_n, feat_n = self.att_n(xL3_n)

            # 拼接并池化
            x = torch.cat((x1_att, x2_att), 0)            # (2B,1024,24,12)

            # 构造全局特征和全局输出（与原 model.py 保持一致）
            feat_globe = torch.cat((feat_v, feat_n), 0)   # (2B,512)
            out_globe = torch.cat((out_v, out_n), 0)      # (2B,class_num)

        elif modal == 1:
            xL1, xL2, xL3 = self.base_resnet(x1)
            x, _, _ = self.att_v(xL3)

        elif modal == 2:
            xL1, xL2, xL3 = self.base_resnet(x2)
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
            for i in range(self.part):
                if i == 0:
                    featf = feat[i]
                else:
                    featf = torch.cat((featf, feat[i]), 1)
            return featf
