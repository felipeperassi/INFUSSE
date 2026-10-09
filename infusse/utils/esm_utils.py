import numpy as np
import os
from pathlib import Path

import torch

import biotite.structure # biotite 1.6 renamed filter_backbone
if not hasattr(biotite.structure, 'filter_backbone'):
    biotite.structure.filter_backbone = biotite.structure.filter_peptide_backbone

import esm
import esm.inverse_folding


def load_structure_encoder(encoder='esmif1'): # Loads a frozen structure encoder
    if encoder == 'esmif1':
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    else:
        raise ValueError(f'Unknown structure encoder: {encoder}')

    for name, param in model.named_parameters():
        param.requires_grad = False # Freezing parameters
    model.eval()

    return model, alphabet

@torch.no_grad()
def encode_backbone(model, alphabet, coords_per_chain, padding_length=10): # H, L & AG = [L, 3, 3] ===ESM-IF1===> [L, 512] (512: embedd dim)
    padding = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    blocks = []
    for i, coords in enumerate(coords_per_chain):
        if i:
            blocks.append(padding)
        blocks.append(np.asarray(coords, dtype=np.float32))

    representation = esm.inverse_folding.util.get_encoder_output(
        model, alphabet, np.concatenate(blocks, axis=0)
    )

    outputs = []
    start = 0
    for i, coords in enumerate(coords_per_chain):
        if i:
            start += padding_length
        outputs.append(representation[start:start+len(coords)])
        start += len(coords)

    return outputs

def save_structure_embeddings(payload, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = file_path.with_suffix(file_path.suffix + '.tmp')
    try:
        torch.save(payload, temporary_file)
        os.replace(temporary_file, file_path)
    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise