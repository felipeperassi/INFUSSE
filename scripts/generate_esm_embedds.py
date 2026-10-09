import argparse
import glob
import numpy as np
import os
import torch

from infusse.config import DATA_DIR, DEFAULT_ENCODER, ENCODER_TYPES, STRUCTURE_EMBEDDING_FILES, STRUCTURE_DIR
from infusse.utils.biology_utils import backbone_coords_from_pdb, cdr_mask_from_pdb, get_first_digit
from infusse.utils.esm_utils import encode_backbone, load_structure_encoder, save_structure_embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', choices=ENCODER_TYPES, default=DEFAULT_ENCODER)
parser.add_argument('--checkpoint_every', type=int, default=100)
args = parser.parse_args()

directory = DATA_DIR
file_list = list(dict.fromkeys(sorted([file for folder in STRUCTURE_DIR for file in glob.glob(os.path.join(folder, '*.pdb')) if '_stripped' not in file], key=get_first_digit)))
pdb_files = {file[-8:-4]: file for file in file_list}
output_file = directory + STRUCTURE_EMBEDDING_FILES[args.encoder]
pdb_codes = np.load(directory+'pdb_codes.npy')
C = torch.load(directory+'chain_inputs.pt')

ag_alone_list = [None] * len(pdb_codes)
complex_list = [None] * len(pdb_codes)
ab_alone_list = [None] * len(pdb_codes)
cdr_list = [None] * len(pdb_codes)

if os.path.exists(output_file):
    cached = torch.load(output_file, map_location='cpu')
    if cached['pdb_codes'] != [str(pdb) for pdb in pdb_codes]:
        raise ValueError(f'Embedding cache does not match {directory}pdb_codes.npy.')
    ag_alone_list = cached['ag_alone']
    complex_list = cached['complex']
    ab_alone_list = cached['ab_alone']
    print(f'Resuming from {output_file}')

cdr_path = directory + 'cdr_mask.pt'
if os.path.exists(cdr_path):
    cdr_list = torch.load(cdr_path)
    
model, alphabet = load_structure_encoder(args.encoder)

for i, pdb in enumerate(pdb_codes):
    if ag_alone_list[i] is not None and complex_list[i] is not None and ab_alone_list[i] is not None and cdr_list[i] is not None:
        continue
    print(f'{i+1}/{len(pdb_codes)} {pdb}')

    h_backbone, l_backbone, ag_backbone = backbone_coords_from_pdb(pdb_files[pdb])
    chain_ids = torch.as_tensor(C[i]).flatten().long()
    n_ag = sum(len(chain) for chain in ag_backbone)
    n_ab = sum(len(chain) for chain in h_backbone + l_backbone)

    # Residue counts must match chain_inputs.pt, or the labels would be misaligned
    if n_ag != int((chain_ids == 2).sum()) or n_ab != int((chain_ids < 2).sum()):
        print('Error')
        print(pdb)
        print(f'AG {n_ag} vs {int((chain_ids == 2).sum())}')
        print(f'AB {n_ab} vs {int((chain_ids < 2).sum())}')
        continue

    cdr = torch.zeros(len(chain_ids), dtype=torch.bool)
    ab_cdr = cdr_mask_from_pdb(pdb_files[pdb])
    if len(ab_cdr) != n_ab:
        print(f'CDR mask length mismatch for {pdb}: {len(ab_cdr)} vs {n_ab}')
        continue
    cdr[chain_ids < 2] = torch.tensor(ab_cdr, dtype=torch.bool)
    cdr_list[i] = cdr

    ag_alone_list[i] = torch.cat(encode_backbone(model, alphabet, ag_backbone), dim=0).half()
    complex_list[i] = torch.cat(encode_backbone(model, alphabet, ag_backbone + h_backbone + l_backbone)[:len(ag_backbone)], dim=0).half()
    ab_alone_list[i] = torch.cat(encode_backbone(model, alphabet, h_backbone + l_backbone), dim=0).half()

    if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
        save_structure_embeddings({
            'pdb_codes': [str(code) for code in pdb_codes],
            'ag_alone': ag_alone_list,
            'complex': complex_list,
            'ab_alone': ab_alone_list,
        }, output_file)

save_structure_embeddings({
    'pdb_codes': [str(code) for code in pdb_codes],
    'ag_alone': ag_alone_list,
    'complex': complex_list,
    'ab_alone': ab_alone_list,
}, output_file)
print(f'Embedded {sum(1 for x in ag_alone_list if x is not None)}/{len(pdb_codes)} complexes')

if any(m is not None for m in cdr_list):
    torch.save(cdr_list, cdr_path)
    print(f'Saved CDR mask for {sum(1 for m in cdr_list if m is not None)}/{len(pdb_codes)} complexes')