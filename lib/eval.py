import numpy as np


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask], minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):

        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))

        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

        return acc, acc_cls, iou, miou, fwavacc

    def dice_score(self, predictions, gts):
        dice_list = []
        for pred, gt in zip(predictions, gts):
            cls_val = np.max(np.unique(gt))
            gt = (gt == cls_val)
            pred = (pred == cls_val)
            overlap = np.sum(gt & pred)
            pred_size = np.sum(pred)
            gt_size = np.sum(gt)
            dice = 2 * overlap / (gt_size + pred_size)
            dice_list.append(dice)
        return np.mean(dice_list)


class Meter(object):
    def __init__(self):
        self.cnt = 0
        self.sum = 0

    def update(self, x):
        self.cnt += 1
        self.sum += x

    @property
    def mean(self):
        if self.cnt == 0:
            return 0
        else:
            return float(self.sum) / self.cnt
