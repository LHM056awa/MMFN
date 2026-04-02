import torch
import torch.nn as nn
import sys
sys.path.append('.')

# ========== Focal Loss 定义 ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=[4.0, 1.0], gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 在 train() 函数内部，原来的 loss_f_rumor = ... 替换为：
# loss_f_rumor = FocalLoss(alpha=[4.0, 1.0], gamma=2.0)
