import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CircleLoss(nn.Module):
    '''multi-label classification loss'''
    def __init__(self, average=True):
        super(CircleLoss, self).__init__()
        self.average = average

    # def forward(self, input, target):
    #     """
    #     :param input:   (batch_size, label_num)
    #     :param target:  (batch_size, label_num)
    #     :return:
    #     """
    #     neg_eles = input * (1-target)
    #     neg_sum = torch.log(1 + torch.sum(torch.exp(neg_eles), 1))
    #     pos_eles = input * target
    #     pos_sum = torch.log(1 + torch.sum(torch.exp(-pos_eles), 1))
    #     _sum = neg_sum + pos_sum
    #     if self.average:
    #         return torch.mean(neg_sum + pos_sum)
    #     else:
    #         return _sum

    def forward(self, y_pred, y_true):
        """
        :param y_true:   (batch_size, label_num)
        :param y_pred:  (batch_size, label_num), 0/1
        :return:
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        total_loss = neg_loss + pos_loss
        if self.average:
            return torch.mean(total_loss)
        else:
            return total_loss


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """

        # not_ignore = target.data.cpu() != self.ignore_index
        # target = target[not_ignore]
        # input = input[not_ignore]

        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size(-1)
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                                 ignore_index=self.ignore_index)


class DSCLoss(nn.Module):
    """DSCLoss: Dice Loss for Data-imbalanced NLP Tasks (Multi-Classification)
    Args:
                smooth: A float number to smooth loss, and avoid NaN error, default: 1
                prediction: Output of Network, a tensor of shape [batch, class_num]
                target: Label of classification, a tensor of shape [batch, ]
    Returns:
                Loss tensor according to args reduction
    Comments:
                Suitable for imbalanced data.
    """

    def __init__(self, smooth=0, ignore_index=-100):
        super(DSCLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        # add by houwei
        prediction = F.softmax(prediction, dim=1)

        not_ignore = target.data.cpu() != self.ignore_index
        target = target[not_ignore]
        prediction = prediction[not_ignore]
        target = F.one_hot(target, num_classes=prediction.size(1))

        num = (1.0 - prediction) * prediction * target + self.smooth
        den = (1.0 - prediction) * prediction + target + self.smooth
        dice = 1.0 - num / den
        # print(dice)

        loss = torch.mean(dice, dim=0)
        # print(loss)
        # 去掉 标签 O 的loss
        loss = loss[1:]
        return loss.mean()


class DiceLoss(nn.Module):
    """DiceLoss: A kind of Dice Loss (Multi-Classification)
    Args:
                smooth: A float number to smooth loss, and avoid NaN error, default: 1
                prediction: Output of Network, a tensor of shape [batch, class_num]
                target: Label of classification, a tensor of shape [batch, ]
    Returns:
                Loss tensor according to args reduction
    """

    def __init__(self, smooth=1, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        prediction = F.softmax(prediction, dim=1)

        not_ignore = target.data.cpu() != self.ignore_index
        target = target[not_ignore]
        prediction = prediction[not_ignore]
        target = F.one_hot(target, num_classes=prediction.size(1))

        num = 2 * prediction * target + self.smooth
        den = prediction.pow(2) + target.pow(2) + self.smooth
        loss = torch.mean(1.0 - num / den, dim=0)
        # print(loss)
        # 去掉 标签 O 的loss
        loss = loss[1:]
        return loss.mean()

        '''
        第一轮 验证
        loss* 1: 0.001245
        loss* 10000: 0.01597
        '''