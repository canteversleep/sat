import torch
from torch import nn
import torch.nn.functional as F
from gnn import GraphNN, GraphReadout, mlp

class VGAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.encoder = VGAEEncoder(input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async)
        self.decoder = VGAEDecoder(latent_size, hidden_size, input_size, mlp_arch)

    def forward(self, data):
        z_mean, z_log_var = self.encoder(data)
        z = self.encoder.reparameterize(z_mean, z_log_var)
        adj_rec = self.decoder(z)
        return z_mean, z_log_var, adj_rec

class VGAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.gnn = GraphNN(input_size, hidden_size, mlp_arch, gnn_iter, gnn_async)
        self.mean_readout = GraphReadout(hidden_size, latent_size, hidden_size)
        self.log_var_readout = GraphReadout(hidden_size, latent_size, hidden_size)
        self.latent_size = latent_size

    def forward(self, data):
        h = self.gnn(data)
        z_mean = self.mean_readout(h[0])
        z_log_var = self.log_var_readout(h[0])
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

class VGAEDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, mlp_arch):
        super().__init__()
        # self.decoder = mlp(latent_size, output_size, mlp_arch['hidden_sizes'], eval('nn.' + mlp_arch['activation'] + '()'), nn.Sigmoid())
        self.decoder = mlp(latent_size, output_size, [hidden_size], eval('nn.' + mlp_arch['activation'] + '()'), nn.Sigmoid())

    def forward(self, z):
        adj_rec = self.decoder(z)
        return adj_rec