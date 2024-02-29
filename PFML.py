import torch
import random
from data_class.FashionMNIST import FashionMNIST
from data_class.CiFAR10 import CiFAR10
from data_class.CiFAR100 import CiFAR100
from data_class.Emnist import Emnist
from client import MetaClient
# from hypernetwork import HyperNetwork
from cnn_mnist import MNISTHyper
from cnn_cifar import CIFARHyper
from collections import OrderedDict
from tensorboardX import SummaryWriter
from cnn_mnist import MNISTcnn
from cnn_cifar import CIFARcnn
from embedding import Embed
import copy
class LPFML:
    def __init__(self, batch_size, qr_split_rate, innerlr, outterlr, innerstep, outterstep, num_client, dataset,
                 n_hidden, embed_dim,embed_lr, hyper_dim, n_kernel,p_rate):
        self.qr_split_rate = qr_split_rate
        self.batch_size = batch_size
        self.innerlr = innerlr
        self.outterlr = outterlr
        self.innerstep = innerstep
        self.outterstep = outterstep
        self.num_client = num_client
        self.dataset = dataset
        self.n_hidden = n_hidden
        self.embed_dim = embed_dim if(embed_dim!=-1)else int(self.num_client+1/4)
        self.embed_lr = embed_lr
        self.hyper_dim = hyper_dim
        self.n_kernel = n_kernel
        self.hyper_lr = 0.05
        self.p_rate = p_rate

        # 不仅训练超网络，还要训练一个共同的网络，超网络生成部分权重为客户端提供个性化部分
        # 数据集和模型可以尝试一下目标检测的
        self.model=None
        self.embed = Embed(self.num_client, self.embed_dim)
        if dataset == "CiFAR10"  :
            self.model = CIFARcnn(out_dim=10)
        elif dataset =="CiFAR100":
            self.model = CIFARcnn(out_dim=100)
        elif dataset == "FashionMNIST" or dataset == "Emnist":
            self.model = MNISTcnn()
        else:
            raise("dataset error!")

        # 获取数据集的同时初始化超网络
        self.usr_sup_set, self.usr_que_set = self.get_dataset()

        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(dev)
        self.hyper_network.to(dev)
        pass


    def get_dataset(self):
        if(self.dataset == "CiFAR10"):
            self.hyper_network = CIFARHyper(self.embed, self.embed_dim, self.hyper_dim, self.n_hidden,
                                            n_kernel = self.n_kernel, outdim=10)
            return CiFAR10(self.batch_size, self.qr_split_rate, self.num_client).get_set()

        if(self.dataset == "CiFAR100"):
            self.hyper_network = CIFARHyper(self.embed, self.embed_dim, self.hyper_dim, self.n_hidden,
                                            n_kernel = self.n_kernel, outdim=100)
            return CiFAR100(self.batch_size, self.qr_split_rate, self.num_client).get_set()

        if(self.dataset == "FashionMNIST"):
            self.hyper_network = MNISTHyper(embed=self.embed,embedding_dim = self.embed_dim)
            return FashionMNIST(self.batch_size, self.qr_split_rate, self.num_client).get_set()

        if(self.dataset == "Emnist"):
            self.hyper_network = MNISTHyper(embed=self.embed,embedding_dim = self.embed_dim)
            return Emnist(self.batch_size, self.qr_split_rate, self.num_client).get_set()


    def run(self):
        ### init optimizer for hypernet ###
        hyper_optimizers = {
            "sgd":torch.optim.SGD(
            [
                {"params":[p for n,p in self.hyper_network.named_parameters() if "embed" not in n]},
                {"params":[p for n,p in self.hyper_network.named_parameters() if "embed" in n], "lr":embed_lr},
            ],lr =self.hyper_lr, momentum=0.9
            ),
            "adam":torch.optim.Adam(params=self.hyper_network.parameters(), lr=self.hyper_lr)
        }

        # embed的优化器
        embed_optimizers = torch.optim.SGD(params=self.embed.parameters(), lr=self.embed_lr)
        hyper_opt = hyper_optimizers["adam"]

        clients = []
        for cid in range(self.num_client):
            c = MetaClient(cid,self.outterlr, self.innerlr,self.innerstep,self.outterstep,self.batch_size,self.n_kernel,self.dataset)
            c.init_dataset(self.usr_sup_set[cid], self.usr_que_set[cid])
            clients.append(c)

        print("succefully create ",self.num_client,"clients")

        writer = SummaryWriter(log_dir="./logs_p_rate")
        for ite in range(num_step):
            total_inner_acc = 0
            total_outter_acc = 0
            total_inner_loss = 0
            total_outter_loss = 0

            # 更新embedding
            client_id = random.choice(range(self.num_client))
            client_id = random.choice(range(self.num_client))
            hyst_dict = self.hyper_network.forward(torch.tensor([client_id],dtype=torch.long))

            ups = []
            ft_ups = []
            models = []
            for c in clients:

                # client这边预置的模型需要写个函数在训练开始前设置，还有训练结束后聚合模型
                outer_model,ft_update,update,inner_acc,inner_loss,outter_acc,outter_loss = c.train(hyst_dict, self.model.state_dict(),self.p_rate)

                # 记录数据
                total_inner_acc+=inner_acc
                total_inner_loss+=inner_loss
                total_outter_acc+=outter_acc
                total_outter_loss+=outter_loss

                ft_ups.append(ft_update)
                ups.append(update)# 这边得要有样本的权重，但是样本的长度权重怎么取,是测试集还是支持集的？
                models.append(outer_model)# 用于outterloop模型的聚合

            embed_optimizers.zero_grad()
            hyper_opt.zero_grad()
            # 计算模型差
            inner_state = OrderedDict({k: tensor.data for k, tensor in hyst_dict.items()})
            for sd in ups:
                delta_theta = OrderedDict({k: hyst_dict[k]-sd[k]/self.num_client for k in inner_state.keys()})
            for sd in ft_ups:
                delta_phi = OrderedDict({k: hyst_dict[k]-sd[k]/self.num_client for k in inner_state.keys()})

            # 计算embed梯度
            embed_grads = torch.autograd.grad(
                list(hyst_dict.values()), self.embed.parameters(), grad_outputs=list(delta_phi.values()),retain_graph=True
            )
            # 计算 phi 梯度
            hnet_grads = torch.autograd.grad(
                list(hyst_dict.values()), self.hyper_network.parameters(), grad_outputs=list(delta_theta.values())
            )

            # 更新hnet权重
            for p,g in zip(self.hyper_network.parameters(), hnet_grads):
                p.grad = g
            for p,g in zip(self.embed.parameters(),embed_grads):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(self.hyper_network.parameters(),50)
            torch.nn.utils.clip_grad_norm_(self.embed.parameters(),50)

            # 更新model权重
            all_weight = OrderedDict()
            for net in models:
                for name_weight in net:
                    if name_weight in all_weight:
                        all_weight[name_weight] += net[name_weight]/self.num_client
                    else:
                        all_weight[name_weight] = net[name_weight]/self.num_client
            self.model.load_state_dict(all_weight)
                

            hyper_opt.step()
            embed_optimizers.step()
            print("--------------第 ",ite," 轮---------------------")
            print("avg inner loop acc:",total_inner_acc/self.num_client)
            print("avg outter loop acc",total_outter_acc/self.num_client)
            print("avg inner loop loss:",total_inner_loss/self.num_client)
            print("avg outter loop loss:",total_outter_loss/self.num_client)
            print("------------------------------------------------")
            writer.add_scalar("avg innerloop acc",total_inner_acc/self.num_client,global_step=ite)
            writer.add_scalar("avg outterloop acc",total_outter_acc/self.num_client,global_step=ite)
            writer.add_scalar("avg innerloop loss",total_inner_loss/self.num_client,global_step=ite)
            writer.add_scalar("avg outterloop loss",total_outter_loss/self.num_client,global_step=ite)
        writer.close()



################################
######## 基本参数
################################
num_step = 600
batch_size = 64
qr_split_rate = 0.2

innerlr = 0.001
outterlr = 0.01
innerstep = 10
outterstep = 10
num_client = 10
################################
######## hypernetwork参数
################################
n_hidden = 3# 隐层个数
embed_dim = -1# embed维度
embed_lr = 0.01# embed向量学习率
hyper_dim = 100# hypernetwork的维度
n_kernel = 16# CNN模型的核心数
p_rate = 0.9

dataset = "FashionMNIST"# FashionMNIST、CiFAR10、CiFAR100
server = LPFML(batch_size, qr_split_rate, innerlr, outterlr, innerstep, outterstep, num_client, dataset,
               n_hidden, embed_dim, embed_lr, hyper_dim, n_kernel,p_rate)
server.run()