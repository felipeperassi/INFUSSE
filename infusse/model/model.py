# -*- coding: utf-8 -*-

r"""This module contains the model class. 

:Authors:   Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""

import torch
from torch_geometric.nn import GCNConv

from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

def compute_diffusion_matrix(L, t):
    eigenvalues, eigenvectors = torch.linalg.eigh(L) 
    diffusion_matrix = eigenvectors @ torch.diag(torch.exp(-t * eigenvalues)) @ eigenvectors.T
    return diffusion_matrix

class DiffGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, t_init=1e-3, normalize=True):
        super().__init__(aggr='add')  
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.t = torch.nn.Parameter(torch.tensor(t_init, dtype=torch.float32))  # t is learnt during training
        self.normalize = normalize  

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        laplacian_edge_index, laplacian_edge_weight = get_laplacian(edge_index, edge_weight, normalization='sym' if self.normalize else None, num_nodes=x.size(0))
        laplacian_dense = to_dense_adj(laplacian_edge_index, edge_attr=laplacian_edge_weight, max_num_nodes=x.size(0))
        diffusion_matrix = compute_diffusion_matrix(laplacian_dense[0], self.t)
        x = torch.matmul(diffusion_matrix, x)
        x = self.lin(x)

        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, lm_dim=1024, lm=None, seq_only=False):
        super().__init__()
        self.lm_dim = lm_dim
        self.lm = lm
        self.seq_only = seq_only
        self.aa_linear = torch.nn.Linear(in_channels, self.lm_dim, bias=False)
        self.c_linear = torch.nn.Linear(in_channels, self.lm_dim, bias=False)
        self.lm_linear = torch.nn.Linear(self.lm_dim, self.lm_dim, bias=True)
        #self.ab_linear = torch.nn.Linear(self.lm_dim, self.lm_dim, bias=True)
        #self.ag_linear = torch.nn.Linear(self.lm_dim, self.lm_dim, bias=True)
        #self.sequence_linear_1 = torch.nn.Linear(self.lm_dim, self.lm_dim, bias=True)
        #self.sequence_linear_2 = torch.nn.Linear(self.lm_dim, out_channels, bias=True)
        self.sequence_linear = torch.nn.Linear(self.lm_dim, out_channels, bias=True)
        #self.conv1 = GCNConv(self.lm_dim, hidden_channels, normalize=True)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        #self.conv3 = GCNConv(hidden_channels, out_channels, normalize=True)
        self.conv1 = DiffGCNLayer(self.lm_dim, hidden_channels)
        #self.conv2 = DiffGCNLayer(hidden_channels, hidden_channels)
        self.conv3 = DiffGCNLayer(hidden_channels, out_channels)

    def forward(self, x, x_out, edge_index, edge_weight=None, c=None):#, len_ab=None, len_ag=0):
        #batch_size = len(len_ag)
        if x_out is None:
            x = self.lm(x)[0]
            x = (self.aa_linear(x) + self.lm_linear(x)).relu()
            x_seq = self.sequence_linear(x)
        else:
            #print(x.shape)
            #mask = x > 4
            #x_out = self.lm(x[None,:].to(torch.int64), output_attentions=False, output_hidden_states=True)['hidden_states'][-1]
            #x_out = x_out[mask.unsqueeze(-1).expand_as(x_out)].view(x_out.size(0), -1, x_out.size(-1))
            #x = x[mask].to(torch.float32).unsqueeze(-1)
            x = (self.c_linear(c.unsqueeze(-1)) + self.aa_linear(x.unsqueeze(-1)) + self.lm_linear(x_out)).relu()
            #x = (torch.cat((self.c_linear(c.unsqueeze(-1)), self.aa_linear(x.unsqueeze(-1)), self.lm_linear(x_out)), dim=-1)).relu()
            x_seq = self.sequence_linear(x)

            '''
            x_processed = []
            start = 0 

            for i in range(batch_size):
                ab_len = len_ab[i]   
                ag_len = len_ag[i] if len_ag is not None else 0 
                
                x_ab = x[start:start + ab_len]
                x_ab = self.ab_linear(x_ab).relu()  

                if ag_len > 0:
                    x_ag = x[start + ab_len:start + ab_len + ag_len]
                    x_ag = self.ag_linear(x_ag).relu()  
                    x_combined = torch.cat((x_ab, x_ag), dim=0) 
                else:
                    x_combined = x_ab  

                x_processed.append(x_combined)
                
                start += ab_len + ag_len

            x = torch.cat(x_processed, dim=0)
            '''
        #x = self.sequence_linear_1(x)
        #x = self.sequence_linear_2(x)
        if not self.seq_only:
            x = self.conv1(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
            #x = self.conv2(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
            x = self.conv3(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)) 
        else:
            x = torch.zeros_like(x_seq, dtype=x_seq.dtype, device=x_seq.device)
        # x_tot = torch.sigmoid(x + x_seq)
        x_tot = x + x_seq
        return x_tot, x