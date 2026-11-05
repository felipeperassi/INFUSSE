# -*- coding: utf-8 -*-

r"""This module contains the model class for epitope prediction. 

:Authors:   Felipe Perassi <fperassi@fi.uba.ar> & Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""

import torch

class SeqStructAttention(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.last_attention = None

    def forward(self, seq_embeddings, struct_embeddings):
        if seq_embeddings.shape[0] == 0 or struct_embeddings.shape[0] == 0:
            return seq_embeddings
        
        struct_attention, weights = self.attention(struct_embeddings, seq_embeddings, seq_embeddings, need_weights=True, average_attn_weights=False)
        self.last_attention = weights
        return self.norm(struct_embeddings + struct_attention) # dim: struct_dim

class AgAbAttention(torch.nn.Module):
    def __init__(self, struct_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=struct_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(struct_dim)
        self.last_attention = None

    def forward(self, ag_embeddings, ab_embeddings):
        if ag_embeddings.shape[0] == 0 or ab_embeddings.shape[0] == 0:
            return ag_embeddings
        
        ag_attention, weights = self.attention(ag_embeddings, ab_embeddings, ab_embeddings, need_weights=True, average_attn_weights=False)
        self.last_attention = weights
        return self.norm(ag_embeddings + ag_attention) # dim: struct_dim

class EpitopeModel(torch.nn.Module):
    def __init__(self, seq_channels, struct_channels, out_channels, hidden_channels, lm_dim=1024, lm=None, seq_only=False):
        super().__init__()
        self.lm_dim = lm_dim
        self.lm = lm
        self.seq_only = seq_only
        self.hidden_channels = hidden_channels # F
        self.struct_channels = struct_channels

        # Sequence block
        self.aa_linear = torch.nn.Linear(seq_channels, self.hidden_channels, bias=False)        # T1
        self.lm_linear = torch.nn.Linear(self.lm_dim, self.hidden_channels, bias=True)          # T2
        self.seq_out_linear = torch.nn.Linear(self.hidden_channels, out_channels, bias=True)    # T3

        # Epitope Graph block
        if self.seq_only:
            self.ag_struct_linear = None
            self.ab_struct_linear = None
            self.ag_ab_att = None
            self.seq_struct_att = None
            self.struct_out_linear = None

        else: 
            self.ag_struct_linear = torch.nn.Linear(self.struct_channels, self.hidden_channels, bias=False) # T ag
            self.ab_struct_linear = torch.nn.Linear(self.struct_channels, self.hidden_channels, bias=False) # T ab
            self.ag_ab_att = AgAbAttention(self.hidden_channels, num_heads=8, dropout=0.3)
            self.seq_struct_att = SeqStructAttention(self.hidden_channels, num_heads=8, dropout=0.3)

            self.struct_out_linear = torch.nn.Linear(self.hidden_channels, out_channels, bias=True)

    def forward(self, x, x_out, x_struct, c=None, cdr_mask=None):
        # x: one-hot [N, seq_channels]
        # x_out: ProtBERT embeddings [N, lm_dim]

        if x_out is None:
            raise ValueError('Sequence embeddings are required.')
        
        if c is None:
            raise ValueError('Chain IDs are required.')

        ag = (c.long() == 2)
        ab = ~ag
        
        # Sequence block
        x_aa = self.aa_linear(x.to(dtype=x_out.dtype)).relu()   # T1: N_one_hot -> F
        x_lm = self.lm_linear(x_out).relu()                     # T2: N_protbert -> F

        x_seq = x_aa + x_lm 
        x_seq_out = self.seq_out_linear(x_seq[ag])              # T3: N_ag -> N_out

        # Structure block
        x_struct_out = torch.zeros_like(x_seq_out)

        if not self.seq_only:
            if x_struct is None or c is None:
                raise ValueError('Structure embeddings and chain IDs are required.')
  
            z_ag = self.ag_struct_linear(x_struct[ag]).relu()   # T_ag: N_ag -> F
            z_ab = self.ab_struct_linear(x_struct[ab]).relu()   # T_ab: N_ab -> F

            if cdr_mask is not None:
                cdr_mask_ab = cdr_mask[ab].bool()
                z_ab = z_ab[cdr_mask_ab]

            z_ag = self.ag_ab_att(z_ag, z_ab)                   # Q = AG, K = V = AB
            
            # z_concat = torch.cat((z_ag, z_ab), dim=0)           # (N_ag + N_ab) -> F
            z = self.seq_struct_att(x_seq, z_ag)            # Q = AG struct, K = V = seq

            n_ag = z_ag.shape[0]
            z_ag = z[:n_ag]
            x_struct_out = self.struct_out_linear(z_ag) # N_ag -> N_out

        x_tot = x_seq_out + x_struct_out
        return x_tot, x_struct_out, x_seq_out