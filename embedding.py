import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch

class Embed(nn.Module):
    def __init__(self, client_num, embeding_dim):
        super().__init__()
        self.embed = nn.Embedding(client_num, embeding_dim)
    def forward(self,client_id):
        client_id = torch.tensor([client_id], dtype=torch.long)
        return self.embed(client_id)
