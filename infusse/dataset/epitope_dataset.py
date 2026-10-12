# -*- coding: utf-8 -*-

r"""This module contains the dataset class for epitope prediction. 

:Authors:   Felipe Perassi <fperassi@fi.uba.ar> & Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

def transformer_embedding(model, tokens, mask, special_token_ids=None):
    def forward(input_ids):
        if special_token_ids is None:
            return model(
                input_ids[None, :].to(torch.int64),
                output_attentions=False,
                output_hidden_states=True,
            )['hidden_states'][-1].squeeze(0)
        return model(input_ids[None, :].to(torch.int64)).last_hidden_state.squeeze(0)

    max_positions = getattr(model.config, 'max_position_embeddings', None)
    if max_positions is None or len(tokens) <= max_positions:
        return forward(tokens)[mask]

    special_ids = torch.as_tensor(special_token_ids, device=tokens.device)
    prefix = tokens[:1] if not mask[0] else tokens[:0]
    suffix = tokens[-1:] if not mask[-1] else tokens[:0]
    chunk_size = max_positions - len(prefix) - len(suffix)
    if chunk_size < 1:
        raise ValueError('Transformer position limit leaves no room for residues.')

    outputs = []
    residue_tokens = tokens[mask]
    for start in range(0, len(residue_tokens), chunk_size):
        chunk = torch.cat((prefix, residue_tokens[start:start + chunk_size], suffix))
        chunk_mask = ~torch.isin(chunk, special_ids)
        outputs.append(forward(chunk)[chunk_mask])
    return torch.cat(outputs)


class EpitopeModelDataset(Dataset):
    def __init__(self, X, Y, device, pdb=None, C=None, lm_ag=None, special_token_ids=None, embeddings=None, struct_embeddings=None, cdr_masks=None):
        self.X = []
        self.X_out = []
        self.num_features = 1
        if embeddings is None:
            print('Generating embeddings with Transformer')
        for i, x in enumerate(X):
            if special_token_ids is None:
                mask = x > 4
            else:
                mask = ~torch.isin(x, torch.as_tensor(special_token_ids, device=x.device))
            if embeddings is None:
                x_out = transformer_embedding(
                    lm_ag, x, mask, special_token_ids=special_token_ids
                )
            else:
                x_out = embeddings[i]
            x = x[mask].to(torch.float32)
            if embeddings is None and ((i + 1) % 100 == 0 or i + 1 == len(X)):
                print(f'Embedded {i + 1}/{len(X)} complexes')
                
            self.X.append(x.to(device).squeeze())
            self.X_out.append(x_out.to(device).squeeze())

        self.C = [torch.Tensor(c).to(device) for c in C]
        self.Y = [y.to(device).float() for y in Y]
        self.pdb = pdb
        self.out_channels = 1

        # Struc embedds
        self.X_struct = None
        self.struct_features = None
        if struct_embeddings is not None:
            paired = isinstance(struct_embeddings, dict)
            ag_list = struct_embeddings['ag'] if paired else struct_embeddings
            ab_list = struct_embeddings['ab'] if paired else None
            self.struct_features = ag_list[0].shape[1]
            self.X_struct = []
            for i, ag in enumerate(ag_list):
                chain_ids = torch.as_tensor(C[i]).flatten().long()
                aligned = ag.new_zeros((len(chain_ids), self.struct_features))
                aligned[chain_ids == 2] = ag
                if paired:
                    aligned[chain_ids < 2] = ab_list[i].to(aligned.dtype)
                self.X_struct.append(aligned.to(device))

        self.cdr_masks = None
        if cdr_masks is not None:
            self.cdr_masks = [m.to(device) for m in cdr_masks]

        for i in range(len(self.X)):
            x_dim = self.X[i].shape[0]
            x_out_dim = self.X_out[i].shape[0]
            y_dim = self.Y[i].shape[0]
            c_dim = len(self.C[i])
            struct_dim = self.X_struct[i].shape[0] if self.X_struct is not None else c_dim
            if not (x_dim == x_out_dim == y_dim == c_dim == struct_dim):
                print(f'Dimension mismatch. PDB {self.pdb[i]}:')
                print(f'  X[{i}] dim: {x_dim}')
                print(f'  X_out[{i}] dim: {x_out_dim}')
                print(f' Y[{i}] dim: {y_dim}')
                print(f'  C[{i}] dim: {c_dim}')
                print(f'  X_struct[{i}] dim: {struct_dim}')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        X_out = self.X_out[idx].to(torch.float32)
        C = self.C[idx]
        Y = self.Y[idx]
        if self.pdb is not None:
            pdb = self.pdb[idx]
        else:
            pdb = None
    
        data = Data(x=X, x_out=X_out, y=Y, c=C, pdb=pdb)
        if self.X_struct is not None:
            data.x_struct = self.X_struct[idx].to(torch.float32)
        if self.cdr_masks is not None:
            data.cdr_mask = self.cdr_masks[idx]
        return data
