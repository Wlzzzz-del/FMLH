import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import copy

# test_data_size = len(test_data)
# train_data_size = len(train_data)


class Base:
    # 所有数据集处理类的抽象类
    def __init__(self, batch_size, q_r_split_rate, client_num):
        self.batch_size = batch_size
        self.q_r_split_rate = q_r_split_rate
        self.client_num = client_num
        # self.train_len,self.test_len = 0
        # self.train_data,self.test_data = None
        # self.train_dataloader, self.test_dataloader = None

        self.init_data()
        self.split_data_to_client()
        self.to_task()


    def init_data(self):
        pass

    def split_data_to_client(self):
        assert(self.train_len!=0 and self.test_len!=0)
        len_per_client_train = int(self.train_len/self.client_num)
        len_per_client_test = int(self.test_len/self.client_num)
        train_dict = dict()
        test_dict = dict()
        for cid in range(self.client_num):
            seg = cid*len_per_client_train
            slicer = slice(seg, seg+(len_per_client_train-1), 1)
            train_dict[cid] = [self.train_data[i] for i in range(*slicer.indices(len(self.train_data)))]
        for cid in range(self.client_num):
            seg = int(cid*len_per_client_test)
            slicer = slice(seg, seg+(len_per_client_train-1), 1)
            test_dict[cid] = [self.test_data[i] for i in range(*slicer.indices(len(self.test_data)))]

        self.all_data = {}
        for cid in range(self.client_num):
            self.all_data[cid] = train_dict[cid]+test_dict[cid]
        del train_dict
        del test_dict

    def to_task(self):
        self.usr_support_set = {}
        self.usr_query_set = {}
        for cid in self.all_data:
            support_len = int(self.q_r_split_rate*(len(self.all_data[cid])))
            self.usr_support_set[cid] = self.all_data[cid][:support_len-1]
            self.usr_query_set[cid] = self.all_data[cid][support_len:]
        print("successfully generate task.")

    def get_set(self):
        return self.usr_support_set,self.usr_query_set