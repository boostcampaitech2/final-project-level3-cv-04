import torchvision.transforms as transforms

from pytorchvideo.transforms import (
    Normalize,
    ShortSideScale
)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
)


x3d_train_transform = transforms.Compose(
                  [
                    Lambda(lambda x: x.permute(3, 0, 1, 2)), # Convert tensor from (T, H, W, C) to (C, T, H, W)"""
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(size=180),
                    CenterCrop((180, 180))
                    # RandomHorizontalFlip(p=0.5),
                  ]
                )
x3d_valid_transform = transforms.Compose(
                [
                    Lambda(lambda x: x.permute(3, 0, 1, 2)),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(size=180),
                    CenterCrop((180, 180))
                ]
                )

transform_entrypoints = {
    'x3d_train' : x3d_train_transform,
    'x3d_valid' : x3d_valid_transform
}

# transform 이름 존재하는지 확인
def is_transform(transform_name):
    return transform_name in transform_entrypoints

# get transform
def get_transform(transform_name):
    if is_transform(transform_name):
        transform = transform_entrypoints[transform_name]
    else:
        raise RuntimeError('Unknown transform (%s)' % transform_name)
    return transform