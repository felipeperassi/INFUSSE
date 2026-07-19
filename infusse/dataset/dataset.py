# -*- coding: utf-8 -*-

r"""This module contains the model class. 

:Authors:   Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from  infusse.utils.biology_utils import separate_tokenised_chains

class GCNBfDataset(Dataset):
    def __init__(self, edge_indices, edge_attributes, X, Y, device, pdb=None, C=None, lm_ab=None, lm_ag=None, special_token_ids=None, embeddings=None):
        self.edge_indices = [edge_index.to(device) for edge_index in edge_indices]
        self.edge_attributes = [edge_attr.to(device) for edge_attr in edge_attributes]
        if lm_ab is None and embeddings is None:
            self.X = [x.to(device) for x in X]
            self.num_features = X[0].shape[1]
        else:
            self.X = []
            self.X_out = []
            self.num_features = 1
            self.len_ab = []
            self.len_ag = []
            if embeddings is None:
                print('Generating embeddings with Transformer')
            for i, x in enumerate(X):
                if special_token_ids is None:
                    mask = x > 4
                else:
                    mask = ~torch.isin(x, torch.as_tensor(special_token_ids, device=x.device))
                if embeddings is None:
                    if special_token_ids is None:
                        x_out = lm_ag(
                            x[None,:].to(torch.int64),
                            output_attentions=False,
                            output_hidden_states=True,
                        )['hidden_states'][-1]
                    else:
                        x_out = lm_ag(x[None,:].to(torch.int64)).last_hidden_state
                    x_out = x_out[mask.unsqueeze(-1).expand_as(x_out)].view(x_out.size(0), -1, x_out.size(-1)).squeeze()
                else:
                    x_out = embeddings[i]
                x = x[mask].to(torch.float32)
                if embeddings is None and ((i + 1) % 100 == 0 or i + 1 == len(X)):
                    print(f'Embedded {i + 1}/{len(X)} complexes')
                '''
                x_ab, x_ag = separate_tokenised_chains(x)
                mask_ab = x_ab > 4 # Special tokens go from 0 to 4 (incl.) in AntiBERTa 
                x_ab = lm_ag(x_ab[None,:].to(torch.int64), output_attentions=False, output_hidden_states=True)['hidden_states'][-1]
                x_ab = x_ab[mask_ab.unsqueeze(-1).expand_as(x_ab)].view(x_ab.size(0), -1, x_ab.size(-1))
                self.len_ab.append(x_ab.size(1))

                if x_ag is not None:
                    mask_ag = x_ag > 4 # Special tokens go from 0 to 4 (incl.) in ProtBERT
                    x_ag = lm_ag(x_ag[None,:].to(torch.int64), output_attentions=False, output_hidden_states=True)['hidden_states'][-1]
                    x_ag = x_ag[mask_ag.unsqueeze(-1).expand_as(x_ag)].view(x_ag.size(0), -1, x_ag.size(-1))
                    x_ab = torch.cat((x_ab, x_ag), axis=1)
                    self.len_ag.append(x_ag.size(1))
                else:
                    self.len_ag.append(0)
                self.X.append(x_ab.to(device).squeeze())
                '''
                self.len_ag.append(x.size(0))
                self.X.append(x.to(device).squeeze())
                self.X_out.append(x_out.to(device).squeeze())

        self.C = [torch.Tensor(c).to(device) for c in C]
        self.Y = [y.to(device) for y in Y]
        self.pdb = pdb
        self.out_channels = 1

        for i in range(len(self.X)):
            x_dim = self.X[i].shape[0]
            x_out_dim = self.X_out[i].shape[0]
            y_dim = self.Y[i].shape[0]
            c_dim = len(self.C[i])
            if not (x_dim == x_out_dim == y_dim == c_dim):
                print(f'Dimension mismatch. PDB {self.pdb[i]}:')
                print(f'  X[{i}] dim: {x_dim}')
                print(f'  X_out[{i}] dim: {x_out_dim}')
                print(f' Y[{i}] dim: {y_dim}')
                print(f'  C[{i}] dim: {c_dim}')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        edge_index = self.edge_indices[idx]
        edge_attr = self.edge_attributes[idx]
        X = self.X[idx]
        X_out = self.X_out[idx]
        #len_ab = self.len_ab[idx]
        len_ag = self.len_ag[idx]
        C = self.C[idx]
        Y = self.Y[idx]
        if self.pdb is not None:
            pdb = self.pdb[idx]
        else:
            pdb = None
    
        return Data(x=X, x_out=X_out, edge_index=edge_index, y=Y, c=C, edge_attr=edge_attr, pdb=pdb, len_ag=len_ag)#, len_ab=len_ab, len_ag=len_ag)
