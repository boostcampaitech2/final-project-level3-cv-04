import torch
import torch.nn as nn
import torch.nn.functional as F

# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal_loss' : FocalLoss
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]

# Loss 이름 존재하는지 확인
def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints

# get Loss
def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion