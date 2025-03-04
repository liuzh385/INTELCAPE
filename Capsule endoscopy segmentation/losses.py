import torch
from itertools import repeat
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
from dice_loss import *

import warnings

warnings.filterwarnings("ignore")


class ST_Loss(nn.Module):
    def __init__(self, n_classes=2, weight=[1, 1]):
        super(ST_Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight
        self.loss_s = SoftDiceLoss()
        self.loss_t = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input_s, input_t, target, texture):
        loss_s = self.loss_s(input_s, target)
        loss_t = self.loss_t(input_t, texture)
        return loss_s * 0.5 + loss_t, loss_t


def get_loss(cfg):
    if cfg.MODEL.DICE_LOSS:
        print("=" * 15, "Using dice_loss", "=" * 15)
        # loss = DiceLossCE(n_classes=cfg.DATA.SEG_CLASSES, weight=cfg.MODEL.WEIGTH)
        # loss = BinaryDiceLoss()
        # loss = SoftDiceLoss(n_classes=cfg.DATA.SEG_CLASSES)
        loss = BSP_dice_loss()
        # loss = GDL()
    elif cfg.MODEL.BCE_LOSS:
        print("=" * 15, "Using BCE_loss", "=" * 15)
        loss = nn.BCEWithLogitsLoss()
    elif cfg.MODEL.CE_LOSS:
        print("=" * 15, "Using CE_loss", "=" * 15)
        loss = nn.CrossEntropyLoss()
    elif cfg.MODEL.L1_LOSS:
        print("=" * 15, "Using L1_loss", "=" * 15)
        loss = nn.L1Loss()
    elif cfg.MODEL.L1_DICE:
        print("=" * 15, "Using L1_loss and dice_loss", "=" * 15)
        loss = [nn.L1Loss(), SoftDiceLoss()]
    return loss


# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]


class BinaryDiceLoss(nn.Module):
    """
        Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, D, H, W].
            logits: a tensor of shape [B, C, D, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
    """

    def __init__(self, smooth=1e-6, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # predict = predict.contiguous().view(predict.shape[0], -1)
        # target = target.contiguous().view(target.shape[0], -1)

        # num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        # loss = 1 - num / den
        target = torch.unsqueeze(target, 1)

        num_classes = predict.shape[1]

        true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()

        probas = F.softmax(predict, dim=1)
        true_1_hot = true_1_hot.type(predict.type())

        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = ((2. * intersection + self.smooth) / (cardinality + self.smooth))
        loss = (1 - dice_loss)
        # print(loss.mean())

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        # print(input.size(), target.size())
        smooth = 1e-8
        batch_size = input.size(0)
        # print(input[100, :, 100, 100], target[100, 100, 100])

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        # input = F.sigmoid(input).view(batch_size, self.n_classes, -1)
        # print(input.max(), input.min())
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        # print(target.max(), target.min())
        # print(input.size(), target.size())
        # print(input[100, :, 34100], target[100, :, 24100])

        inter = torch.sum(input * target, 2) + smooth
        # print(torch.sum(input * target, 2).mean())
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
        # print(torch.sum(input, 2).mean(), torch.sum(target, 2).mean())

        # score = torch.mean(2.0 * inter / union)
        score = 2.0 * torch.mean(inter) / torch.mean(union)
        # print(score)
        # score = 1.0 - (score / (float(batch_size) * float(self.n_classes)))
        score = 1.0 - score
        # print(score)

        return score


class CustomSoftDiceLoss(nn.Module):
    def __init__(self, class_ids, n_classes=2):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 1e-6
        batch_size = input.size(0)

        input = F.softmax(input[:, self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, n_classes=2, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.one_hot_encoder = One_Hot(n_classes).forward

    def forward(self, predict, target):
        target = self.one_hot_encoder(target).contiguous()
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda()

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class DiceLossCE(nn.Module):

    def __init__(self, n_classes=2, weight=1.0, ignore_index=None, **kwargs):
        super(DiceLossCE, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, output, target):
        loss = self.weight * F.cross_entropy(output, target)
        # print('loss_ce', loss)
        output = F.softmax(output, dim=1)
        for c in [1, 2, 3]:
            o = output[:, c]
            t = (target == c).float()
            loss += 0.25 * self.dice_per_im(o, t)
            # print('loss ', c, loss)

        return loss

    def dice_per_im(self, output, target):
        eps = 1e-8
        n = output.shape[0]
        output = output.view(n, -1)
        target = target.view(n, -1)
        num = 2 * (output * target).sum(1) + eps
        den = output.sum(1) + target.sum(1) + eps
        return 1.0 - (num / den).mean()


class BSP_dice_loss(nn.Module):
    def __init__(self):
        super(BSP_dice_loss, self).__init__()

    def forward(self, input, target):
        """soft dice loss"""
        eps = 1e-7
        iflat = input.view(-1)
        tflat = target.view(-1)
        # print(iflat.max(), iflat.min(), tflat.max(), tflat.min())
        intersection = (iflat * tflat).sum()

        loss = 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

        # print('loss === >', loss)

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


if __name__ == '__main__':
    # from torch.autograd import Variable
    # depth=3
    # batch_size=2
    # encoder = One_Hot(depth=depth).forward
    # y = Variable(torch.LongTensor(batch_size, 1, 1, 2 ,2).random_() % depth).cuda()  # 4 classes,1x3x3 img
    # y_onehot = encoder(y)
    # x = Variable(torch.randn(y_onehot.size()).float()).cuda()
    # dicemetric = SoftDiceLoss(n_classes=depth)
    # dicemetric(x,y)

    out = torch.rand((5, 2, 80, 80, 80), dtype=torch.float)
    target = torch.zeros((5, 80, 80, 80), dtype=torch.bool)

    loss = SoftDiceLoss()

    loss.forward(out, target)
