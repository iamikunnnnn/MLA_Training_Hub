import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
def get_dataset():
    Fashion_MNIST_train = datasets.MNIST(root='./dataset', train=True, download=True,
                                                transform=transforms.ToTensor())
    Fashion_MNIST_test = datasets.MNIST(root='./dataset', train=False, download=True,
                                           transform=transforms.ToTensor())
    return Fashion_MNIST_train,Fashion_MNIST_test

get_dataset()