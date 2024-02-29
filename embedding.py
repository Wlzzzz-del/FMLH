import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch

class Embed(nn.Module):
    def __init__(self, client_num, embeding_dim):
        super().__init__()
        # embeding_dim = int(client_num+1/4)
        self.embed = nn.Embedding(client_num, embeding_dim)
    def forward(self,client_id):
        # 先转成tensor
        client_id = torch.tensor([client_id], dtype=torch.long)
        # 输出
        return self.embed(client_id)