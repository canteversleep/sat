import torch
from torch import nn
import torch.nn.functional as F
from gnn import GraphNN, GraphReadout, mlp, GraphNNFHead
import numpy as np
from collections import defaultdict



######### PyTorch Geometric Implementation -- Strong VGAE ########

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops



        
####### Weak VGAEs ought to predict SAT class alongside other characterizing features of the SAT problem ########

class WeakVGAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.encoder = VGAEEncoder(input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async)
        self.decoder = VGAEDecoder(latent_size, hidden_size, input_size, mlp_arch)

    def forward(self, data): # TODO: CHK1 retun this to be variational once you figure out classification
        z_mean, z_log_var = self.encoder(data)
        z = self.encoder.reparameterize(z_mean, z_log_var)
        class_pred = self.decoder(z)
        # h = self.encoder(data)
        # class_pred = self.decoder(h)
        return 0, 0, class_pred

class VGAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.gnn = GraphNN(input_size, hidden_size, mlp_arch, gnn_iter, gnn_async)
        self.mean_readout = GraphReadout(hidden_size, latent_size, hidden_size)
        self.log_var_readout = GraphReadout(hidden_size, latent_size, hidden_size)
        # self.latent = nn.Sequential(
        #     nn.Linear(hidden_size, latent_size),
        #     nn.ReLU()
        # )
        self.latent_size = latent_size

    def forward(self, data):
        h = self.gnn(data)
        h_pool = torch.mean(torch.cat(h), dim=0)
        # h1_pool = torch.mean(h[1], dim=0)
        # latent = self.latent(h_pool)
        z_mean = self.mean_readout(h_pool)
        z_log_var = self.log_var_readout(h_pool)
        return z_mean, z_log_var
        # return latent

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

class VGAEDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, mlp_arch):
        super().__init__()
        self.n_classes = 4
        # we want to predict the number of variables, clauses, and the number of positive literals
        # self.n_properties = 3
        # self.decoder = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.Linear(latent_size, self.n_classes)
        # self.property_retriever = GraphReadout(hidden_size, self.n_properties, hidden_size)
        
    def forward(self, z):
        # print(f'z shape: {z.shape}')
        logits = self.decoder(z)
        # return F.softmax(logits, dim=-1)
        return logits
        # class_pred = self.class_redout(h)
        

class VGAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super(VGAE, self).__init__()
        self.encoder = GraphEncoder(input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async)
        self.decoder = DotProductDecoder()

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, data):
        mean, log_var = self.encoder(data)
        zv = self.reparameterize(mean[0], log_var[0])
        zc = self.reparameterize(mean[1], log_var[1])
        adj_reconstructed = self.decoder(zv,zc)
        return adj_reconstructed, mean, log_var

class GraphEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super(GraphEncoder, self).__init__()
        self.gnn = GraphNN(input_size, hidden_size, mlp_arch, gnn_iter, gnn_async)
        self.mean_gnn = GraphNNFHead(hidden_size, latent_size, {"hidden_sizes": [32], "activation": False}, gnn_iter, gnn_async)
        self.log_var_gnn = GraphNNFHead(hidden_size, latent_size, {"hidden_sizes": [32], "activation": False}, gnn_iter, gnn_async)

    def forward(self, data):
        h = self.gnn(data)
        # return h
        mean = self.mean_gnn(h, data)
        log_var = self.log_var_gnn(h, data)
        return (mean, log_var)  # Assuming we are using the variable nodes for encoding


class DotProductDecoder(nn.Module):
    def forward(self, zv, zc):
        adj = torch.sigmoid(torch.matmul(zv, zc.t()))
        return adj
