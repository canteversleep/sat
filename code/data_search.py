import os
import pdb
import random
from collections import namedtuple

import scipy.sparse as sparse
import torch
import numpy as np

from cnf import CNF
from util import DataSample, adj, adj_batch, init_edge_attr, to_sparse_tensor
from torch_geometric.utils import to_undirected


Batch = namedtuple('Batch', ['x', 'adj', 'sol'])


def load_dir(path):
    data = []
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext != '.cnf':
            continue
        f = CNF.from_file(os.path.join(path, filename))
        if name.startswith('uu'):
            continue
        data.append(DataSample(filename, f, adj(f), None))
    return data


def init_tensors(sample, device):
    # [1,0,0] -> assigned False
    # [0,1,0] -> assigned True
    # [0,0,1] -> clause
    adj = sample.adj
    n, m = adj[0].shape[0], adj[0].shape[1]
    xv = torch.zeros(n, 3, dtype=torch.float32)
    sol = [x if random.random() < 0.5 else -x for x in range(n + 1)]
    xv[torch.arange(n), (torch.tensor(sol[1:]) > 0).long()] = 1
    xv = xv.to(device)
    xc = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(m, 1).to(device)
    xev = init_edge_attr(n).to(device)
    xec = init_edge_attr(m).to(device)
    vadj = to_sparse_tensor(sparse.hstack(adj)).to(device)
    cadj = to_sparse_tensor(sparse.vstack(adj)).t().to(device)
    return Batch((xv, xc, xev, xec), (vadj, cadj), sol)


# def vgae_wrapper(batch, device):
#     (xv, xc, xev, xec), (vadj, cadj), sol = batch
    
#     # Stack the node features (xv and xc)
#     x = torch.cat((xv, xc), dim=0).to(device)
    
#     # Convert the sparse adjacency matrices to edge indices
#     v_edge_index = torch.tensor(np.vstack((vadj.coalesce().indices())), dtype=torch.long).to(device)
#     c_edge_index = torch.tensor(np.vstack((cadj.coalesce().indices())), dtype=torch.long).to(device)
    
#     # Combine the edge indices
#     edge_index = torch.cat((v_edge_index, c_edge_index), dim=1)
    
#     # Convert the sparse adjacency matrices to dense adjacency matrix
#     adj = torch.tensor((vadj + cadj).todense(), dtype=torch.float).to(device)
    
#     return x, edge_index, adj


def init_tensors_vgae(sample, device):
    # [1,0,0] -> assigned False
    # [0,1,0] -> assigned True
    # [0,0,1] -> clause
    adj = sample.adj
    n, m = adj[0].shape[0], adj[0].shape[1]
    
    # Create node features
    xv = torch.zeros(n, 3, dtype=torch.float32)
    sol = [x if random.random() < 0.5 else -x for x in range(n + 1)]
    xv[torch.arange(n), (torch.tensor(sol[1:]) > 0).long()] = 1
    xv = xv.to(device)
    xc = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(m, 1).to(device)
    x = torch.cat((xv, xc), dim=0)
    
    # Create adjacency matrix using variable-clause adjacency
    adj_var_clause = to_sparse_tensor(sparse.hstack(adj)).to(device)
    
    # Create edge index
    edge_index = torch.tensor(np.vstack((adj_var_clause.coalesce().indices())), dtype=torch.long).to(device)
    # edge_index = to_undirected(edge_index)
    
    return x, edge_index, adj_var_clause.to_dense()

def init_tensors_vgae_naive(sample, device):
    # [1,0,0] -> assigned False
    # [0,1,0] -> assigned True
    # [0,0,1] -> clause
    adj_pos, adj_neg = sample.adj
    n, m = adj_pos.shape
    
    # Create node features
    xv = torch.zeros(n, 3, dtype=torch.float32)
    sol = [x if random.random() < 0.5 else -x for x in range(n + 1)]
    xv[torch.arange(n), (torch.tensor(sol[1:]) > 0).long()] = 1
    xv = xv.to(device)
    xc = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(m, 1).to(device)
    x = torch.cat((xv, xc), dim=0)
    
    # Create adjacency matrix
    adj = adj_pos + adj_neg
    adj = to_sparse_tensor(adj).to(device)
    
    # Create edge index
    edge_index = torch.tensor(np.vstack((adj.coalesce().indices())), dtype=torch.long).to(device)
    
    return x, edge_index, adj.to_dense()


def init_tensors_vgae_naive_v1(sample, device):
    # [1,0,0] -> assigned False
    # [0,1,0] -> assigned True
    # [0,0,1] -> clause
    adj_pos, adj_neg = sample.adj
    n, m = adj_pos.shape
    
    # Create node features
    xv = torch.zeros(n, 3, dtype=torch.float32)
    sol = [x if random.random() < 0.5 else -x for x in range(n + 1)]
    xv[torch.arange(n), (torch.tensor(sol[1:]) > 0).long()] = 1
    xv = xv.to(device)
    xc = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(m, 1).to(device)
    x = torch.cat((xv, xc), dim=0)
    
    # Create adjacency matrix
    adj = adj_pos + adj_neg
    adj = to_sparse_tensor(adj).to(device)
    
    # Create edge indices
    edge_index_var_to_clause = adj.coalesce().indices()
    edge_index_clause_to_var = torch.stack((adj.coalesce().indices()[1], adj.coalesce().indices()[0]))
    edge_index = torch.cat((edge_index_var_to_clause, edge_index_clause_to_var), dim=1).to(device)
    
    return x, edge_index, adj.to_dense()


def init_tensors_vgae_bipartite(sample, device):
    # [1,0,0] -> assigned False
    # [0,1,0] -> assigned True
    # [0,0,1] -> clause
    adj_pos, adj_neg = sample.adj
    n, m = adj_pos.shape
    
    # Create node features
    xv = torch.zeros(n, 3, dtype=torch.float32)
    sol = [x if random.random() < 0.5 else -x for x in range(n + 1)]
    xv[torch.arange(n), (torch.tensor(sol[1:]) > 0).long()] = 1
    xv = xv.to(device)
    xc = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(m, 1).to(device)
    x = torch.cat((xv, xc), dim=0)
    
    # Create adjacency matrices
    adj_var_to_clause = to_sparse_tensor(adj_pos + adj_neg).to(device)
    adj_clause_to_var = adj_var_to_clause.t()
    
    # Create edge indices
    edge_index_var_to_clause = adj_var_to_clause.coalesce().indices()
    edge_index_clause_to_var = adj_clause_to_var.coalesce().indices()
    edge_index = torch.cat((edge_index_var_to_clause, edge_index_clause_to_var), dim=1).to(device)
    
    return x, edge_index, adj_var_to_clause.to_dense()

def vgae_wrapper(batch, device):
    (xv, xc, xev, xec), (vadj, cadj), sol = batch
    
    # Stack the node features (xv and xc)
    x = torch.cat((xv, xc), dim=0).to(device)
    
    # Convert the sparse adjacency matrices to edge indices
    v_edge_index = torch.tensor(np.vstack((vadj.coalesce().indices())), dtype=torch.long).to(device)
    c_edge_index = torch.tensor(np.vstack((cadj.coalesce().indices())), dtype=torch.long).to(device)
    
    # Combine the edge indices
    edge_index = torch.cat((v_edge_index, c_edge_index), dim=1)
    
    # Convert the sparse adjacency matrices to dense adjacency matrix
    adj = torch.tensor((vadj + cadj.transpose(0, 1)).todense(), dtype=torch.float).to(device)
    
    return x, edge_index, adj