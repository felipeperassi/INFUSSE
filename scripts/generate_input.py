import argparse
import glob
import numpy as np
import os
import scipy
import torch
import re

from infusse.config import DATA_DIR, STRUCTURE_DIR
from infusse.utils.biology_utils import encode_line, generate_one_hot_matrix, get_first_digit, get_tokenised_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--cssp', type=bool, default=False)
parser.add_argument('--plm', choices=['protbert', 'antiberta2', 'ankh'], default='protbert')
args = parser.parse_args()

directory = DATA_DIR
input_folder = directory + 'strides_outputs/'
file_list = list(dict.fromkeys(sorted([file for folder in STRUCTURE_DIR for file in glob.glob(os.path.join(folder, '*.pdb')) if '_stripped' not in file], key=get_first_digit)))
output_folder = directory + 'gcn_inputs/'
pdb_codes = np.load(directory+'pdb_codes.npy')
X_list = []
C_list = []
X_ab_list = []
print(len(pdb_codes))

for file in file_list:
    if file[-8:-4] in pdb_codes:
        if args.lm == 'transformer':
            X, C, X_ab = get_tokenised_sequence(file, args.cssp, args.plm)
            X_list.append(X)
            C_list.append(C)
            X_ab_list.append(X_ab)
        else:
            X = generate_one_hot_matrix(input_folder+file[-8:-4]+'.txt')
            X_list.append(torch.from_numpy(X))

input_name = 'gcn_inputs.pt' if args.plm == 'protbert' else f'gcn_inputs_{args.plm}.pt'
torch.save(X_list, directory+input_name)
torch.save(C_list, directory+'chain_inputs.pt')
torch.save(X_ab_list, directory+'sequences.pt')
