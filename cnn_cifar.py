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


class CIFARHyper(nn.Module):
    def __init__(self,embed, embedding_dim,hidden_dim=10,in_channels=3,channels=32, outdim=10, n_hidden=1):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.outdim = outdim

        # embedding_dim = int(client_num+1/4)
        # self.embeddings = nn.Embedding(client_num, embedding_dim=embedding_dim)
        self.embeddings = embed

        # 构造layers生成hidden_dim, embedding_dim---->hidden_dim
        layers = [
            nn.Linear(embedding_dim, hidden_dim)
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, self.channels*self.in_channels*3*3)
        self.c1_bias = nn.Linear(hidden_dim, channels)
        self.c2_weights = nn.Linear(hidden_dim, 2*self.channels*self.channels*3*3)
        self.c2_bias = nn.Linear(hidden_dim, 2*channels)
        self.fc1_weights = nn.Linear(hidden_dim, 2*self.channels * 8 * 8*128)
        self.fc1_bias = nn.Linear(hidden_dim, 128)
        self.fc2_weights = nn.Linear(hidden_dim, 128*10)
        self.fc2_bias = nn.Linear(hidden_dim, 10)

    def forward(self, cid):
        emd = self.embeddings(cid)
        features = self.mlp(emd)
        weights = OrderedDict({
            "conv1.weight":self.c1_weights(features).view(self.channels, self.in_channels, 3, 3),
            "conv1.bias":self.c1_bias(features).view(-1),
            "conv2.weight":self.c2_weights(features).view(2*self.channels,self.channels, 3, 3),
            "conv2.bias":self.c2_bias(features).view(-1),
            "fc1.weight":self.fc1_weights(features).view(128, 2*self.channels * 8 * 8),
            "fc1.bias":self.fc1_bias(features).view(-1),
            "fc2.weight":self.fc2_weights(features).view(self.outdim,128),
            "fc2.bias":self.fc2_bias(features).view(self.outdim)
        })
        return weights

class CIFARcnn(nn.Module):
    def __init__(self,in_channels=3, channels=32, out_dim = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(channels, 2*channels, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2*channels * 8 * 8, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# hyper = CIFARHyper()
# node_id = random.choice(range(10))
# state_dict = hyper(torch.tensor([1],dtype=torch.long))

# model = CIFARcnn(in_channels=3,channels=32, out_dim=10)
# model.load_state_dict(state_dict)
# # state_dict = model.state_dict()
# # for i in model.state_dict():
# #     print(i,"的size为:",state_dict[i].shape)

# train_dataset = CIFAR10(root='./CIFAR10_dataset', train=True, download=True, transform=Compose([ToTensor(),Normalize(mean=(0.5,),std=(0.5,))]))
# # train_dataset = FashionMNIST(root='path/to/FashionMNIST', train=True, download=True, transform=Compose([ToTensor(),Normalize(mean=(0.5,),std=(0.5,))]))
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# for inputs,labels in train_loader:
#     # print(inputs[0])
#     model(inputs)
