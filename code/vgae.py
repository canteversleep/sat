import torch
from torch import nn
import torch.nn.functional as F
from gnn import GraphNN, GraphReadout, mlp
import numpy as np


######### PyTorch Geometric Implementation -- Strong VGAE ########

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, args):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(args.enc_in_channels,
                                                          args.enc_hidden_channels,
                                                          args.enc_out_channels),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, all_edge_index):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss + kl_loss

        
####### Weak VGAEs ought to predict SAT class alongside other characterizing features of the SAT problem ########

class WeakVGAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.encoder = VGAEEncoder(input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async)
        self.decoder = VGAEDecoder(latent_size, hidden_size, input_size, mlp_arch)

    def forward(self, data):
        z_mean, z_log_var = self.encoder(data)
        z = self.encoder.reparameterize(z_mean, z_log_var)
        print(z.shape)
        adj_rec = self.decoder(z)
        return z_mean, z_log_var, adj_rec

class VGAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.gnn = GraphNN(input_size, hidden_size, mlp_arch, gnn_iter, gnn_async)
        self.mean_readout = GraphReadout(hidden_size, latent_size, hidden_size)
        self.log_var_readout = GraphReadout(hidden_size, latent_size, hidden_size)
        # self.mean_readout = GraphNN(hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async)
        # self.log_var_readout = GraphNN(hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async)
        self.latent_size = latent_size

    def forward(self, data):
        h = self.gnn(data)
        h_pool = torch.mean(h, dim=1)
        z_mean = self.mean_readout(h_pool)
        z_log_var = self.log_var_readout(h_pool)
        print(z_mean.shape, z_log_var.shape)
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

        
class MultiHeadVGAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.encoder = VGAEEncoder(input_size, hidden_size, latent_size, mlp_arch, gnn_iter, gnn_async)
        self.node_decoder = NodeFeatureDecoder(latent_size, hidden_size, input_size, mlp_arch)
        self.adj_pos_decoder = AdjacencyDecoder(latent_size, hidden_size, mlp_arch)
        self.adj_neg_decoder = AdjacencyDecoder(latent_size, hidden_size, mlp_arch)

    def forward(self, data):
        z_mean, z_log_var = self.encoder(data)
        z = self.encoder.reparameterize(z_mean, z_log_var)
        node_features_rec = self.node_decoder(z)
        adj_pos_rec = self.adj_pos_decoder(z, node_features_rec)
        adj_neg_rec = self.adj_neg_decoder(z, node_features_rec)
        return z_mean, z_log_var, node_features_rec, adj_pos_rec, adj_neg_rec


class NodeFeatureDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, mlp_arch):
        super().__init__()
        self.decoder = mlp(latent_size, output_size, [hidden_size],
                           eval('nn.' + mlp_arch['activation'] + '()'),
                           final_activation=nn.Sigmoid())  # Use Sigmoid if the output is binary

    def forward(self, z):
        node_features = self.decoder(z)
        return node_features


class AdjacencyDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, mlp_arch):
        super().__init__()
        # Assuming output_size is computed or set externally based on graph size
        output_size = computed_output_size  # This needs to be defined based on your graph's characteristics
        self.decoder = mlp(latent_size, output_size, [hidden_size],
                           eval('nn.' + mlp_arch['activation'] + '()'),
                           final_activation=nn.Sigmoid())  # Sigmoid to represent probabilities of edges

    def forward(self, z, node_features):
        # Optionally use node_features to assist in reconstructing the adjacency matrix
        combined_input = torch.cat([z, node_features], dim=-1)
        adj_matrix = self.decoder(combined_input)
        return adj_matrix.reshape(-1, num_nodes, num_nodes)  # Reshape output to matrix form
