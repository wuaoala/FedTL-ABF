from torch import nn
from sklearn import metrics
import torch
from .data_preprocess import *
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predict, target):
        # pt = nn.functional.sigmoid(predict)[:, 1] # sigmoid获取概率    sigmoid激活函数 + BCE损失函数
        pt = nn.functional.softmax(predict, dim=-1)[:, 1]  # softmax获取概率 CE损失函数无需softmax
        pred = predict.max(dim=1, keepdim=True)[1]
        # 在原始ce上增加动态权重因子
        # 根据不平衡比率设定权重
        # y_1_num = torch.nonzero(target == 1).shape[0]
        # y_0_alpha = y_1_num / target.shape[0]
        # y_1_alpha = 1 - y_0_alpha
        # loss = - y_1_alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
        #        - y_0_alpha * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        y_0_num = torch.nonzero(target == 0).shape[0]
        y_1_num = torch.nonzero(target == 1).shape[0]
        y_0_ration = y_0_num / target.shape[0]
        y_1_alpha = y_0_ration / (1 - y_0_ration)

        confusion = metrics.confusion_matrix(target.detach().numpy(), pred.detach().numpy())
        FN = confusion[1][0]
        TN = confusion[0][0]
        TP = confusion[1][1]
        FP = confusion[0][1]

        alpha = (y_0_num + TP) / (y_1_num + TP)
        # alpha = (y_0_num + TP) / (y_1_num + TP)
        # alpha = (y_0_num + (TP * rec)) / (y_1_num + (TP * rec))
        y_1_alpha = alpha
        # loss = - y_1_alpha * (1 - pt) * target * torch.log(pt) \
        #        - pt * (1 - target) * torch.log(1 - pt)

        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        return torch.mean(loss)

class WCELoss(nn.Module):
    def __init__(self, alpha=0.75):
        super(WCELoss, self).__init__()
        self.alpha = alpha

    def forward(self, predict, target):
        pt = nn.functional.softmax(predict, dim=-1)[:, 1]  # softmax获取概率 CE损失函数无需softmax
        loss = - self.alpha * target * torch.log(pt) - (1 - self.alpha) * (1 - target) * torch.log(1 - pt)
        return torch.mean(loss)

class GHMLoss(nn.Module):
    def __init__(self, bins=5, momentum=0.5):
        super(GHMLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(0, 1.001, 1.0 / bins)
        self.edges[-1] += 1e-6
        self.edges = self.edges.cpu()
        self.acc_sum = torch.zeros(bins).cpu()
        self.loss_sum = torch.zeros(bins).cpu()

    def forward(self, logits, targets):
        edges = self.edges.detach()
        mmt = self.momentum
        weights = torch.zeros_like(logits.sigmoid()[:, 1])
        g = torch.abs(logits.sigmoid()[:, 1].detach() - targets).view(-1)

        total = logits.numel()
        acc_sum = self.acc_sum
        loss_sum = self.loss_sum
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            acc_sum[i] = mmt * acc_sum[i] + (1 - mmt) * num_in_bin
            weights.view(-1)[inds] = total / acc_sum[i]

        loss = nn.BCEWithLogitsLoss(weight=weights)(logits[:, 1], targets.to(dtype=float))
        # loss = nn.CrossEntropyLoss(weight=weights)(logits[:,1], targets.to(dtype=float))
        return loss

class Hingeloss(nn.Module):
    def __init__(self):
        super(Hingeloss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = 2 * y_true - 1
        # pt = nn.functional.softmax(y_pred, dim=-1)[:, 1]
        pt = y_pred[:, 1]
        loss = torch.mean(torch.clamp(1 - pt * y_true, min=0))
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        input = nn.functional.softmax(input, dim=1)[:, 1]
        smooth = 1
        intersection = input * target
        loss = 1 - (2 * torch.sum(intersection) + smooth) / (torch.sum(input) + torch.sum(target) + smooth)
        return loss.mean()

def CS_loss(loss_name):

    if loss_name == 'FocalLoss':
        criterion = FocalLoss()
    elif loss_name == 'GHMLoss':
        criterion = GHMLoss()
    elif loss_name == 'Hingeloss':
        criterion = Hingeloss()
    elif loss_name == 'WCELoss':
        criterion = WCELoss()

    return criterion

