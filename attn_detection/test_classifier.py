import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image

DATA_DIR = "data/"


def predict_image(image, test_transforms, device, model):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_ = Variable(image_tensor)
    input_ = input_.to(device)
    output = model(input_)
    index = output.data.cpu().numpy().argmax()
    return index


def get_random_images(num, test_transforms):
    data = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels, classes


def test():
    test_transforms = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('attention_model.pth')
    model.eval()

    to_pil = transforms.ToPILImage()
    images, labels, classes = get_random_images(5, test_transforms)
    fig = plt.figure(figsize=(10, 10))
    for ii in range(len(images)):
        image = to_pil(images[ii])
        index = predict_image(image, test_transforms, device, model)
        sub = fig.add_subplot(1, len(images), ii+1)
        res = int(labels[ii]) == index
        sub.set_title(str(classes[index]) + ":" + str(res))
        plt.axis('off')
        plt.imshow(image)
    plt.show()


test()
