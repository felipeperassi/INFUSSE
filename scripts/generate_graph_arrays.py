import argparse
import glob
import numpy as np
import os
import scipy
import torch

from torch_geometric.utils.convert import from_scipy_sparse_matrix

from infusse.config import ADJACENCIES_DIR, CM_DIR, DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', type=str, default='gnm')
parser.add_argument('--force_unweighted', action='store_true')
args = parser.parse_args()

if args.graphs == 'gnm':
    input_folder = CM_DIR
elif args.graphs == 'bagpype':
    input_folder = ADJACENCIES_DIR
elif args.graphs == 'wgnm':
    input_folder = WGNM_DIR

directory = DATA_DIR
pdb_codes = np.load(directory+'pdb_codes.npy')
ei_list = []
ea_list = []
inv_laplacian_list = []
X = torch.load(directory+'gcn_inputs.pt')
Y = torch.load(directory+'ab_ag_labels.pt')

for i, pdb in enumerate(pdb_codes):
    adjacency = scipy.sparse.load_npz(input_folder+pdb+'.npz')

    if args.force_unweighted:
        thresholded_adjacency = adjacency.copy()
        thresholded_adjacency.data = (thresholded_adjacency.data != 0).astype(int)
        edge_index, edge_attr = from_scipy_sparse_matrix(thresholded_adjacency)
    else:
        edge_index, edge_attr = from_scipy_sparse_matrix(adjacency)
    edge_attr = edge_attr.clone().to(torch.float)
    if edge_index.shape[-1] != edge_attr.shape[0] or Y[i].shape[0] != adjacency.toarray().shape[0] or X[i].shape[0] not in range(adjacency.toarray().shape[0]-10, adjacency.toarray().shape[0]+10):
        print('Error')
        print(pdb)
        print(X[i].shape[0])        
        print(Y[i].shape[0])
        print(adjacency.toarray().shape[0])
    ei_list.append(edge_index)
    ea_list.append(edge_attr)
torch.save({'edge_index': ei_list, 'edge_attr': ea_list}, directory+'edge_data.pt')
