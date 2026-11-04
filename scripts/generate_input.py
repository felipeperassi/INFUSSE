import argparse
import glob
import numpy as np
import os
import scipy
import torch
import re

from infusse.config import DATA_DIR, STRUCTURE_DIR
from infusse.utils.biology_utils import encode_line, generate_one_hot_matrix, get_first_digit, get_tokenised_sequence, get_transformer_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--cssp', type=bool, default=False)
parser.add_argument('--plm', choices=['protbert', 'antiberta2', 'ankh'], default='protbert')
args = parser.parse_args()

if args.lm == 'onehot':
    if args.plm != 'protbert':
        parser.error('This one-hot conversion currently expects ProtBERT inputs.')

    X_tokens = torch.load(DATA_DIR + 'gcn_inputs.pt', map_location='cpu', weights_only=True)
    C = torch.load(DATA_DIR + 'chain_inputs.pt', map_location='cpu', weights_only=True)
    pdb_codes = np.load(DATA_DIR + 'pdb_codes.npy')

    if not (len(X_tokens) == len(C) == len(pdb_codes)):
        raise ValueError('Input files contain different numbers of complexes.')

    tokeniser = get_transformer_tokenizer('protbert')

    # Same category order as generate_one_hot_matrix, using one-letter names
    amino_acids = [
        ' ', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W',
        'C', 'P', 'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A',
        'B', 'Z', 'R', 'M'
    ]
    aa_mapping = {aa: i for i, aa in enumerate(amino_acids)}

    X_list = []
    for i, tokens in enumerate(X_tokens):
        tokens = tokens[tokens > 4].long() # Same filtering as the dataset

        if len(tokens) != len(C[i]):
            raise ValueError(f'Residue count mismatch: {pdb_codes[i]}')

        residues = tokeniser.convert_ids_to_tokens(tokens.tolist())

        if any(aa not in aa_mapping for aa in residues):
            raise ValueError(f'Unrecognised residue token: {pdb_codes[i]}')

        indices = torch.tensor(
            [aa_mapping[aa] for aa in residues],
            dtype=torch.long,
        )

        X = torch.nn.functional.one_hot(
            indices, num_classes=len(amino_acids)
        ).float()

        X_list.append(X)

    output_file = DATA_DIR + 'gcn_inputs_onehot.pt'
    torch.save({
        'pdb_codes': [str(pdb) for pdb in pdb_codes],
        'amino_acids': amino_acids,
        'one_hot': X_list,
    }, output_file)

    print(f'Saved one-hot inputs for {len(X_list)} complexes: {output_file}')
    raise SystemExit

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
