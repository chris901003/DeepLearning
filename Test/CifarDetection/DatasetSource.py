from torchvision import datasets


def cifar100_dataset_from_official(root='cifar100', train=True, download=True, transform_data=None):
    data = datasets.CIFAR100(
        root=root, train=train, download=download, transform=transform_data
    )
    return data
