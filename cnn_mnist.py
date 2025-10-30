from collections import OrderedDict
import random
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor,Normalize,Compose



class MNISTHyper(nn.Module):
    def __init__(self,embed, embedding_dim, hidden_dim=10,in_channels=1,channels=16, outdim=10, n_hidden=1):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.outdim = outdim
        self.embeddings = embed

        layers = [
            nn.Linear(embedding_dim, hidden_dim)
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, self.channels*self.in_channels*5*5)
        self.c1_bias = nn.Linear(hidden_dim, channels)
        self.c2_weights = nn.Linear(hidden_dim, 2*self.channels*self.channels*5*5)
        self.c2_bias = nn.Linear(hidden_dim, 2*channels)
        self.fc1_weights = nn.Linear(hidden_dim, 2*self.channels * 7 * 7*128)
        self.fc1_bias = nn.Linear(hidden_dim, 128)
        self.fc2_weights = nn.Linear(hidden_dim, 128*10)
        self.fc2_bias = nn.Linear(hidden_dim, 10)

    def forward(self, cid):
        emd = self.embeddings(cid)
        features = self.mlp(emd)
        weights = OrderedDict({
            "conv1.weight":self.c1_weights(features).view(self.channels, self.in_channels, 5, 5),
            "conv1.bias":self.c1_bias(features).view(-1),
            "conv2.weight":self.c2_weights(features).view(2*self.channels,self.channels, 5, 5),
            "conv2.bias":self.c2_bias(features).view(-1),
            "fc1.weight":self.fc1_weights(features).view(128, 2*self.channels * 7 * 7),
            "fc1.bias":self.fc1_bias(features).view(-1),
            "fc2.weight":self.fc2_weights(features).view(self.outdim,128),
            "fc2.bias":self.fc2_bias(features).view(self.outdim)
        })
        return weights

class MNISTcnn(nn.Module):
    def __init__(self, in_channels=1,channels=16, outdim=10):
        super().__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 定义全连接层
        self.fc1 = nn.Linear(2*channels * 7 * 7, 128)
        self.fc2 = nn.Linear(128, outdim)
        # 定义激活函数
        self.relu = nn.ReLU()
    def forward(self, x):
        # 输入 x 的 shape 为 (batch_size, 1, 28, 28)
        out = self.conv1(x)  # shape：(batch_size, 16, 28, 28)
        out = self.relu(out)
        out = self.pool1(out)  # shape：(batch_size, 16, 14, 14)
        out = self.conv2(out)  # shape：(batch_size, 32, 14, 14)
        out = self.relu(out)
        out = self.pool2(out)  # shape：(batch_size, 32, 7, 7)
        out = out.view(-1,32*7*7)  # 将张量展开为一维，以便进行全连接
        out = self.fc1(out)  # shape：(batch_size, 128)
        out = self.relu(out)
        out = self.fc2(out)  # shape：(batch_size, 10)

        return out


