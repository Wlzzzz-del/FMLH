from collections import OrderedDict
from cnn_mnist import FashionMNIST
from cnn_cifar import CIFARHyper
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch
import random

class HyperNetwork(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        # spectral_norm进行谱归一化
        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]

        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        # python中*可迭代对象表示解压该对象
        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)# 根据idx生成embedding
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2 * self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })
        return weights


### ---------------MAIN HERE----------------
# n_hidden = 3
# embed_dim = 3# -1
# hyper_hid = 100
# nkernels = 16
# num_nodes = 10# client数
# out_dim = 10
# hypernet = CNNHyper(num_nodes,embed_dim,in_channels=3,hidden_dim=hyper_hid,n_hidden=n_hidden)

# node_id = random.choice(range(num_nodes))
# weights = hypernet(torch.tensor([node_id], dtype=torch.long))
# print(weights)
# cnn = CNNTarget(3,16,10)
# cnn.load_state_dict(weights)
