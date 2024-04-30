import itertools
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from gnn import GraphNN, GraphReadout



class PoolingGraphEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.gnn = GraphNN(input_size, hidden_size, mlp_arch, gnn_iter, gnn_async)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Adaptive average pooling
        self.latent_readout = nn.Linear(hidden_size, latent_size)
        self.latent_size = latent_size

    def forward(self, data):
        h = self.gnn(data)
        pooled = self.pool(h[0].transpose(1, 2)).squeeze(2)
        latent = self.latent_readout(pooled)
        return latent

class GraphEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.gnn = GraphNN(input_size, hidden_size, mlp_arch, gnn_iter, gnn_async)
        self.latent_readout = GraphReadout(hidden_size, latent_size, hidden_size)
        self.latent_size = latent_size

    def forward(self, data):
        h = self.gnn(data)
        latent = self.latent_readout(h[0])
        return latent

class GraphDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, latent):
        adj_rec = self.decoder(latent)
        return adj_rec

class GAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.encoder = GraphEncoder(input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async)
        self.decoder = GraphDecoder(latent_size, hidden_size, input_size)

    def forward(self, data):
        latent = self.encoder(data)
        adj_rec = self.decoder(latent)
        return adj_rec

    def encode(self, data):
        return self.encoder(data)