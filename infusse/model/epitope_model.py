# -*- coding: utf-8 -*-

r"""This module contains the model class for epitope prediction. 

:Authors:   Felipe Perassi <fperassi@fi.uba.ar> & Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""

import torch

class SeqStructMerge(torch.nn.Module): # [ proj. seq embedd (hidden_dim) | proj. struct embedd (hidden_dim) ] -> [ merged embedd ] 
    def __init__(self, seq_dim, struct_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.seq_projection = torch.nn.Sequential(
            torch.nn.Linear(seq_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )
        self.struct_projection = torch.nn.Sequential(
            torch.nn.Linear(struct_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )
        self.merge = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        # self.merge = torch.nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, seq_embeddings, struct_embeddings):
        seq_proj = self.seq_projection(seq_embeddings)
        struct_proj = self.struct_projection(struct_embeddings)
        return self.merge(torch.cat((seq_proj, struct_proj), dim=-1)) # dim: hidden_dim

class SeqStructAttention(torch.nn.Module):
    def __init__(self, seq_dim, struct_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=struct_dim, kdim=seq_dim, vdim=seq_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(struct_dim)

    def forward(self, seq_embeddings, struct_embeddings):
        if seq_embeddings.shape[0] == 0 or struct_embeddings.shape[0] == 0:
            return seq_embeddings
        
        struct_attention, _ = self.attention(struct_embeddings, seq_embeddings, seq_embeddings, need_weights=False)
        return self.norm(struct_embeddings + struct_attention) # dim: struct_dim

class AgAbAttention(torch.nn.Module): # Q = AG, K = V = AB
    def __init__(self, struct_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=struct_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(struct_dim)

    def forward(self, ag_embeddings, ab_embeddings):
        if ag_embeddings.shape[0] == 0 or ab_embeddings.shape[0] == 0:
            return ag_embeddings
        
        ag_attention, _ = self.attention(ag_embeddings, ab_embeddings, ab_embeddings, need_weights=False)
        return self.norm(ag_embeddings + ag_attention) # dim: struct_dim

class EpitopeModel(torch.nn.Module):
    def __init__(self, seq_channels, struct_channels, out_channels, lm_dim=1024, lm=None, seq_only=False):
        super().__init__()
        self.lm_dim = lm_dim
        self.lm = lm
        self.seq_only = seq_only
        
        # Sequence block
        self.aa_linear = torch.nn.Linear(seq_channels, self.lm_dim, bias=False)
        self.c_linear = torch.nn.Linear(seq_channels, self.lm_dim, bias=False)
        self.lm_linear = torch.nn.Linear(self.lm_dim, self.lm_dim, bias=True)
        self.seq_out_linear = torch.nn.Linear(self.lm_dim, out_channels, bias=True)

        # Epitope Graph block
        self.struct_channels = struct_channels
        
        if self.seq_only:
            self.seq_struct_ag_att = None
            self.seq_struct_ab_att = None
            self.ag_att = None
            self.struct_out_linear = None

        else: 
            self.seq_struct_ag_att = SeqStructAttention(self.lm_dim, self.struct_channels, num_heads=8, dropout=0.3)
            self.seq_struct_ab_att = SeqStructAttention(self.lm_dim, self.struct_channels, num_heads=8, dropout=0.3)

            self.ag_att = AgAbAttention(self.struct_channels, num_heads=8, dropout=0.3)
            self.struct_out_linear = torch.nn.Linear(self.struct_channels, out_channels, bias=True)

    def forward(self, x, x_out, x_struct, c=None, cdr_mask=None):
        # Sequence block
        if x_out is None:
            x = self.lm(x)[0]
            x = (self.aa_linear(x) + self.lm_linear(x)).relu()
        else:
            x = (self.c_linear(c.unsqueeze(-1)) + self.aa_linear(x.unsqueeze(-1)) + self.lm_linear(x_out)).relu()
        
        x_seq_out = self.seq_out_linear(x)

        # Epitope Graph block
        if self.seq_only:
            x_struct_out = torch.zeros_like(x_seq_out)

        else:
            ag = (c.long() == 2)
            ab = ~ag

            z_ag = self.seq_struct_ag_att(x[ag], x_struct[ag])
            z_ab = self.seq_struct_ab_att(x[ab], x_struct[ab])

            if cdr_mask is not None:
                cdr_mask_ab = cdr_mask[ab]
                z_ab = z_ab[cdr_mask_ab]

            z_ag = self.ag_att(z_ag, z_ab)

            x_struct_out = torch.zeros_like(x_seq_out)
            x_struct_out[ag] = self.struct_out_linear(z_ag)

        x_tot = x_seq_out + x_struct_out
        return x_tot, x_struct_out, x_seq_out