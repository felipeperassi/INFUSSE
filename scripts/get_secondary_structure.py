import argparse
import glob
import numpy as np
import os
import pickle
import scipy
import torch
import re

from infusse.config import CHECKPOINTS_DIR, DATA_DIR, DEFAULT_GRAPH, GRAPH_TYPES, STRUCTURE_DIR
from infusse.utils.biology_utils import generate_secondary
from infusse.utils.torch_utils import get_dataloaders, load_transformer_weights

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', choices=GRAPH_TYPES, default=DEFAULT_GRAPH)
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--hidden_channels', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

input_folder = DATA_DIR + 'strides_outputs/'
secondary_list = []

checkpoint_path = CHECKPOINTS_DIR + f'{args.graphs}_{args.lm}_features_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}/'

lm = load_transformer_weights(family='general')
lm_ab = load_transformer_weights(family='antibody', cssp=False)
train_loader, test_loader, test_size, dataset = get_dataloaders(
    checkpoint_path,
    device=device,
    lm_ab=lm_ab,
    lm_ag=lm,
    graph_type=args.graphs,
)

for j, loader in enumerate(test_loader):
    path = os.path.join(STRUCTURE_DIR, loader.pdb[0]+'.pdb')
    print(path[-8:-4])
    if os.path.exists(path):  
        secondary = generate_secondary(input_folder+path[-8:-4]+'.txt')
        secondary_list.append(secondary)

with open(DATA_DIR+'secondary.pkl', 'wb') as f:
    pickle.dump(secondary_list, f)
