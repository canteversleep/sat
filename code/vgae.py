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
        print(z.shape)
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
        print(len(h))
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
