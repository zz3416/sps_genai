"""Data-loading helpers for the CIFAR-10 and VAE practicals."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(
    data_dir,
    batch_size=32,
    train=True,
    num_workers=0,
    download=True,
    dataset="cifar10",
):
    """Create a CIFAR-10 or padded Fashion-MNIST data loader.

    The PDF shows ``data/train`` and ``data/test`` in its example. TorchVision
    stores both CIFAR-10 splits beneath one root, so either path is accepted and
    normalized to its parent ``data`` directory.
    """
    from pathlib import Path

    data_path = Path(data_dir)
    root = data_path.parent if data_path.name.lower() in {"train", "test"} else data_path
    dataset_name = dataset.replace("-", "").replace("_", "").lower()
    if dataset_name == "cifar10":
        dataset_object = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
    elif dataset_name == "fashionmnist":
        dataset_object = datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose(
                [transforms.Pad(2), transforms.ToTensor()]
            ),
        )
    else:
        raise ValueError("dataset must be 'cifar10' or 'fashion_mnist'")

    return DataLoader(
        dataset_object,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,
    )
