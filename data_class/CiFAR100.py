from data_class.Base import Base
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

class CiFAR100(Base):
    def __init__(self, batch_size, q_r_split_rate, client_num):
        super().__init__(batch_size, q_r_split_rate, client_num)
    def init_data(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.train_data = torchvision.datasets.CIFAR100(root="CIFAR10_dataset",train=True,download=True,transform=transform)
        self.test_data = torchvision.datasets.CIFAR100(root="CIFAR10_dataset",train=False,download=True,transform=transform)
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        # self.train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
        # self.test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        print("successfully read CiFAR100, train data len:",self.train_len, " test_len:", self.test_len)

# e = CiFAR100(64,0.2,10)