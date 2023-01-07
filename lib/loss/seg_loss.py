import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ["SegLoss", "BCE","ClsLoss"]


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, lbl):
        loss = self.loss(pred, lbl)
        return loss

class ClsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=2, alpha=[0.05, 1, 2])
        self.bce_loss = nn.CrossEntropyLoss()
        self.soft_miou_loss = SoftIoULoss(3)

    def forward(self, outputs, labels, iou_meter=None):
        pred_id = outputs
        _, cls_id = labels
        # focal_loss = self.focal_loss(seg_maps, seg_lbl)
        bce_loss = self.bce_loss(pred_id, cls_id)
        # iou_loss = self.soft_miou_loss(seg_maps, seg_lbl, iou_meter=iou_meter)
        loss = 0.1 * bce_loss

        return loss


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=2, alpha=[0.05, 1, 2])
        self.bce_loss = nn.CrossEntropyLoss()
        self.soft_miou_loss = SoftIoULoss(3)

    def forward(self, outputs, labels, iou_meter=None):
        seg_maps, pred_id = outputs
        seg_lbl, cls_id = labels
        focal_loss = self.focal_loss(seg_maps, seg_lbl)
        bce_loss = self.bce_loss(pred_id, cls_id)
        iou_loss = self.soft_miou_loss(seg_maps, seg_lbl, iou_meter=iou_meter)
        loss = 1.0 * (focal_loss + iou_loss) + 0.1 * bce_loss

        return loss


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).cuda().scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target, iou_meter=None):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter_ = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter_.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - inter_
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        if iou_meter:
            iou_meter.update(loss.mean().item())

        # Return average loss over classes and batch
        return 1-loss.mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return F.cross_entropy(output, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, x, gt):
        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1)  # N,C,H,W => N,C,H*W
            x = x.transpose(1, 2)    # N,C,H*W => N,H*W,C
            x = x.contiguous().view(-1, x.size(2))   # N,H*W,C => N*H*W,C
        gt = gt.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, gt)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != x.data.type():
                self.alpha = self.alpha.type_as(x.data)
            at = self.alpha.gather(0, gt.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
