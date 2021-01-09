import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models


def split_data(data_path, valid_size=0.2):
    train_transforms = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), ])
    test_transforms = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), ])
    train_data = datasets.ImageFolder(data_path, transform=train_transforms)
    test_data = datasets.ImageFolder(data_path, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(
        test_data, sampler=test_sampler, batch_size=64)
    return trainloader, testloader


DATA_DIR = "/data"
trainloader, testloader = split_data(DATA_DIR)
print(trainloader.dataset.classes)
