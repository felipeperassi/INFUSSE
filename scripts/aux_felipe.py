import torch, numpy as np
from infusse.config import DATA_DIR
from infusse.utils.torch_utils import max_chain_identity

X = torch.load(DATA_DIR + 'gcn_inputs.pt')
C = torch.load(DATA_DIR + 'chain_inputs.pt')
pdb = list(np.load(DATA_DIR + 'pdb_codes.npy'))
a, b = pdb.index('1g7i'), pdb.index('1c08')

print('len X:', len(X[a]), 'len C:', len(C[a]))          # <- si difieren, ahi esta
print('identidad que reporta:', max_chain_identity(X, C, a, b))   # deberia dar 1.0