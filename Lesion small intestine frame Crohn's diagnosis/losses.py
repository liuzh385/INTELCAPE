import torch
import numpy as np
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


# -*- coding: utf-8 -*-
# @Author  : LG
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=0.45, gamma=2, num_classes=2, size_average=True):
        '''
            focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
            步骤详细的实现了 focal_loss损失函数.
            :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
            :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
            :param num_classes:     类别数量
            :param size_average:    损失计算方式,默认取均值
        '''
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        '''
            focal_loss损失计算
            :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
            :param labels:  实际类别. size:[B,N] or [B]
            :return:
        '''
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def get_loss(cfg):
    if cfg.MODEL.BCE_LOSS:
        print("=" * 15, "Using BCE_loss", "=" * 15)
        loss = nn.BCEWithLogitsLoss()
    elif cfg.MODEL.CE_LOSS:
        print("=" * 15, "Using CE_loss", "=" * 15)
        loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.25, 1.0])).float())
        # loss = nn.CrossEntropyLoss()
    elif cfg.MODEL.FOCAL_LOSS:
        print("=" * 15, "Using FOCAL_loss", "=" * 15)
        loss = focal_loss()
    elif cfg.MODEL.L1_LOSS:
        print("=" * 15, "Using L1_loss", "=" * 15)
        loss = nn.L1Loss()
    else:
        raise KeyError('Not choose loss function')
    return loss
