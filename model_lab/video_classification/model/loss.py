import torch.nn as nn

_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss
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