import torch
import torch.nn as nn

class cross_center_loss(nn.Module):
    def __init__(self, margin=0, dist_type='l2', start_epoch=20, shared_weight=1.0, cross_weight=1.0):
        super(cross_center_loss, self).__init__()
        self.margin = margin
        self.start_epoch = start_epoch
        self.dist_type = dist_type
        self.shared_weight = shared_weight
        self.cross_weight = cross_weight
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        elif dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        elif dist_type == 'l1':
            self.dist = nn.L1Loss()
        else:
            raise ValueError("Unsupported distance type")

    def forward(self, feat1, feat2, label1, label2, epoch):
        loss = 0
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)

        for i in range(label_num):
            center_v = torch.mean(feat1[i], dim=0)
            center_i = torch.mean(feat2[i], dim=0)
            center_s = (center_v + center_i) / 2

            loss_shared = (max(0, self.dist(center_v, center_s)-self.margin) + max(0, self.dist(center_i, center_s)-self.margin)) / 2

            if epoch < self.start_epoch:
                loss += self.shared_weight * loss_shared
            else:
                feat_num = feat1[i].size(0)
                feat1_single = feat1[i].chunk(feat_num, 0)
                feat2_single = feat2[i].chunk(feat_num, 0)
                loss_cross_v = 0
                loss_cross_i = 0

                for j in range(feat_num):
                    loss_cross_v += max(0, self.dist(torch.squeeze(feat1_single[j]), center_i)-self.margin)
                    loss_cross_i += max(0, self.dist(torch.squeeze(feat2_single[j]), center_v)-self.margin)

                loss_cross = (loss_cross_v + loss_cross_i) / (2 * feat_num)
                loss += self.shared_weight * loss_shared + self.cross_weight * loss_cross

        return loss / label_num
