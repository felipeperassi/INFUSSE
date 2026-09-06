# -*- coding: utf-8 -*-

r"""Training, perturbation and evaluation utilities for the perturbation sweep.

Kept separate from torch_utils.py so the original INFUSSE code stays byte-identical
and the unperturbed baseline can always be reproduced exactly.

:Authors: Felipe Perassi
"""

import inspect

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from infusse.dataset.dataset import GCNBfDataset
from infusse.config import VAL_SEED

def apply_perturb(batch, perturb, params=None, seed=None):
    """seed=None -> global RNG: a fresh, independent mask on every call (training).
    seed=int  -> deterministic (validation, test, and test replicates).

    The inspect check is because identity/remove_off_diagonal take no seed argument.
    """
    if perturb is None:
        return batch.edge_index, batch.edge_attr
    params = dict(params or {})
    if 'seed' in inspect.signature(perturb).parameters:
        params['seed'] = seed
    return perturb(batch.edge_index, batch.edge_attr, batch.c, **params)


def _confusion(y, pred):
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def _rates(tp, fp, fn, tn):
    """Threshold-dependent metrics: the four AsEP reports, plus accuracy kept only
    for continuity with the earlier results."""
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        'mcc': float((tp * tn - fp * fn) / den) if den > 0 else 0.0,
        'f1': f1, 'precision': prec, 'recall': rec,
        'accuracy': (tp + tn) / n if n else 0.0,   # read next to prevalence, never alone
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def compute_metrics(logits, labels, threshold=0.5):
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(int)
    prevalence = float(y.mean())
    auprc = float(average_precision_score(y, prob))

    out = _rates(*_confusion(y, (prob > threshold).astype(int)))
    out.update({
        'auroc': float(roc_auc_score(y, prob)),   # AsEP, PEPNet
        'auprc': auprc,                           # DiscoTope, ScanNet 
        'auprc_over_chance': auprc / max(prevalence, 1e-12),
        # context, not metrics: nothing above is interpretable without these
        'n': int(y.size),
        'prevalence': prevalence,   # chance level: 1-p for accuracy, p for AUPRC
        'threshold': float(threshold),
    })
    return out


def pick_threshold(logits, labels, objective='mcc'):
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(int)
    grid = np.unique(np.quantile(prob, np.linspace(0.001, 0.999, 400)))
    best_t, best_v = 0.5, -2.0
    for t in grid:
        v = _rates(*_confusion(y, (prob > t).astype(int)))[objective]
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


def make_split_loaders(path, device, lm_ab=None, lm_ag=None):
    edge_data = torch.load(path + 'edge_data.pt')
    X = torch.load(path + 'gcn_inputs.pt')
    Y = torch.load(path + 'ab_ag_labels.pt')
    C = torch.load(path + 'chain_inputs.pt')
    pdb_codes = np.load(path + 'pdb_codes.npy')

    dataset = GCNBfDataset(edge_data['edge_index'], edge_data['edge_attr'], X, Y,
                           device=device, pdb=pdb_codes, C=C, lm_ab=lm_ab, lm_ag=lm_ag)

    test_idx = set(np.load(path + 'test_indices.npy').tolist())
    val_idx = set(np.load(path + 'val_indices.npy').tolist())
    overlap = test_idx & val_idx
    if overlap:   # silent leakage would look perfectly normal in the results
        raise ValueError(f'val and test overlap in {len(overlap)} entries: {sorted(overlap)[:10]}')

    train_idx = [i for i in range(len(dataset)) if i not in test_idx and i not in val_idx]
    print(f'train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}')

    return (DataLoader([dataset[i] for i in train_idx], batch_size=1, shuffle=True),
            DataLoader([dataset[i] for i in sorted(val_idx)], batch_size=1),
            DataLoader([dataset[i] for i in sorted(test_idx)], batch_size=1),
            dataset)


def train_perturbed(model, optimiser, train_loader, train_size,
                    perturb=None, perturb_params=None):
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        # seed=None -> global RNG: independent mask per complex and per epoch.
        edge_index, edge_attr = apply_perturb(batch, perturb, perturb_params, seed=None)

        optimiser.zero_grad()
        out, _ = model(batch.x, batch.x_out, edge_index, edge_attr, batch.c)
        ag = (batch.c.long() == 2)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=out.device))(
            torch.squeeze(out)[ag], torch.squeeze(batch.y)[ag])
        tr_loss += batch.num_graphs * loss.item() / train_size
        loss.backward()
        optimiser.step()
    return float(tr_loss)


@torch.no_grad()
def evaluate(model, loader, perturb=None, perturb_params=None,
             threshold=0.5, seed=VAL_SEED):
    """Returns metrics plus raw logits, so any other metric can be recomputed later
    without retraining. Subsumes the old test_perturb: pass perturb= to reproduce it."""
    model.eval()
    logits, labels, pdbs, sizes = [], [], [], []
    for batch in loader:
        edge_index, edge_attr = apply_perturb(batch, perturb, perturb_params, seed=seed)
        pred = model(batch.x, batch.x_out, edge_index, edge_attr, batch.c)[0]
        ag = (batch.c.long() == 2)
        logits.append(torch.squeeze(pred)[ag])
        labels.append(torch.squeeze(batch.y)[ag])
        pdbs.append(str(batch.pdb[0] if isinstance(batch.pdb, (list, tuple)) else batch.pdb))
        sizes.append(int(ag.sum()))
    logits, labels = torch.cat(logits), torch.cat(labels)
    return compute_metrics(logits, labels, threshold), logits, labels, pdbs, sizes