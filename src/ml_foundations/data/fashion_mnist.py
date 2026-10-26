from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ..models.constants import DEFAULT_IMAGE_SIZE

DEFAULT_BATCH_SIZE = 128
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 0
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)
TEST_LIMIT_RATIO = 5
VAL_SPLIT_RATIO = 0.1


def make_loaders(root="data", batch_size=DEFAULT_BATCH_SIZE, seed=DEFAULT_SEED,
                 num_workers=DEFAULT_NUM_WORKERS, download=True, limits=None,
                 image_size=DEFAULT_IMAGE_SIZE):
    operations = []
    if image_size != DEFAULT_IMAGE_SIZE:
        operations.append(transforms.Resize(image_size))
    operations.extend([transforms.ToTensor(), transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD)])
    transform = transforms.Compose(operations)
    train = datasets.FashionMNIST(root, train=True, download=download, transform=transform)
    test = datasets.FashionMNIST(root, train=False, download=download, transform=transform)
    if limits:
        train = torch.utils.data.Subset(train, range(min(limits, len(train))))
        test = torch.utils.data.Subset(test, range(min(limits // TEST_LIMIT_RATIO or 1, len(test))))
    n_val = max(1, int(VAL_SPLIT_RATIO * len(train)))
    n_train = len(train) - n_val
    generator = torch.Generator().manual_seed(seed)
    tr, va = random_split(train,[n_train, n_val],generator=generator)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True,
                   generator=generator, num_workers=num_workers),
        DataLoader(va, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )
