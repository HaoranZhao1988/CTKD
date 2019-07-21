import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def mnist_get_datasets(data_dir):
    """
    Load the MNIST dataset.
    """
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    train_transform = transforms.Compose([ transforms.RandomHorizontalFlip(), \
                                           transforms.ToTensor(), \
                                           normalize ])
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=train_transform, download=True)

    test_transform = transforms.Compose([ transforms.ToTensor(), normalize ])
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=test_transform, download=True)

    return train_dataset, test_dataset


def tinyimagenet_get_datasets(data_dir):
    ds = []
    """
    Load the Tiny ImageNet dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) # using mean and std of original imagenet dataset

    print('reading data..')
    # using 5 crop of image for training
    # train_transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.FiveCrop(64),
    #     transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
    #     #normalize,
    # ])


    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    ds.append(train_dataset)
    test_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)
    ds.append(test_dataset)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

