import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_utils import Decoder, HashDecoder

class ColorNetwork(nn.Module):
    def __init__(
        self,
        cfg,
        num_layers=1,
        hidden_dim=32,
        hash_max_res=2048,
        hash_num_levels=16
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.net = HashDecoder(3, self.hidden_dim, 3, self.num_layers, max_res=hash_max_res, num_levels=hash_num_levels)
    
    def forward(self, x):
        albedo = torch.sigmoid(self.net(x))
        return albedo