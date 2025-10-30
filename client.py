import torch
import torch.nn as nn
import copy
from collections import OrderedDict
from cnn_mnist import MNISTcnn
from cnn_cifar import CIFARcnn
from torch.utils.data import DataLoader

class MetaClient:
    def __init__(self, cid, outterlr, innerlr, inner_step, outter_step, batch_size, n_kernel,  dataset):
        self.cid = cid
        self.outterlr = outterlr
        self.innerlr = innerlr
        self.outter_step = outter_step
        self.inner_step = inner_step
        self.batch_size = batch_size
        self.critierion = nn.CrossEntropyLoss()

        if dataset == "CiFAR10"  :
            self.model = CIFARcnn(out_dim=10)
        elif dataset =="CiFAR100":
            self.model = CIFARcnn(out_dim=100)
        elif dataset == "FashionMNIST" or dataset == "Emnist":
            self.model = MNISTcnn()
        else:
            raise("dataset error!")


        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model.to(dev)
        pass

    def init_dataset(self, sup_data, que_data):
        self.sup_loader = DataLoader(sup_data,self.batch_size,True)
        self.que_loader = DataLoader(que_data,self.batch_size,True)

    def train(self, state_dict):
        self.model.load_state_dict(state_dict)

        grads,inner_acc,inner_loss,suptotal_sample,ft_update = self.finetune()# innerloop
        self.load_grads(grads)
        update,outter_acc,outter_loss,quetotal_sample = self.personalize()# outterloop
        return [ft_update,update,inner_acc,inner_loss,outter_acc,outter_loss]

    def finetune(self):
        # innerloop
        fast_model = copy.deepcopy(self.model)
        inner_opt = torch.optim.Adam(fast_model.parameters(),lr=self.innerlr)
        total_loss = 0
        total_correct = 0
        total_sample = 0
        grads = []
        for iite in range(self.inner_step):
            inner_opt.zero_grad()

            img, label = next(iter(self.sup_loader))
            pred = fast_model(img)
            loss = self.critierion(pred,label)
            total_loss += loss.item()
            total_correct += pred.argmax(1).eq(label).sum().item()
            total_sample += len(label)
            loss.backward()
            inner_opt.step()
            grads += [p.grad for p in fast_model.parameters() if p.grad is not None]
        ft_update = fast_model.state_dict()
        total_correct =total_correct/ total_sample
        return grads,total_correct,total_loss,total_sample,ft_update

    def personalize(self):
        # outterloop
        outter_opt = torch.optim.Adam(self.model.parameters(), lr = self.outterlr)
        outter_opt.step()
        total_loss=0
        total_correct = 0
        total_sample = 0
        for oite in range(self.outter_step):
            img, label = next(iter(self.que_loader))
            pred = self.model(img)
            loss = self.critierion(pred, label)# entropy内部
            total_loss += loss.item()
            total_correct += pred.argmax(1).eq(label).sum().item()
            total_sample += len(label)
            loss.backward()
            outter_opt.step()

        state_dict = self.model.state_dict()
        total_correct =total_correct/ total_sample
        return state_dict, total_correct, total_loss, total_sample

    def load_grads(self,grads):
        for i, p in enumerate(self.model.parameters()):
            if p.grad is not None:
                p.grad += grads[i]
