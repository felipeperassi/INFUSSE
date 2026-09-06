import argparse
import json
import os
import time

import numpy as np
import torch
import csv

from infusse.config import CHECKPOINTS_DIR, DATA_DIR, SEED
from infusse.model.model import GCN
from infusse.utils import perturbations as P
from infusse.utils.experiments import (evaluate, make_split_loaders, pick_threshold,
                                       train_perturbed)
from infusse.utils.torch_utils import load_transformer_weights   # read-only, unmodified

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--patience', type=int, default=6)
parser.add_argument('--only', type=str, default=None, help='run a single config by name')
args = parser.parse_args()

CONFIGS = {
    'clean':    dict(perturb=None),
    'seq_only': dict(perturb=None, seq_only=True),
    'remove':   dict(perturb=P.remove_off_diagonal),
    'shuffle':  dict(perturb=P.off_diagonal_shuffle, eval_replicates=5),
    'drop_p50': dict(perturb=P.dropout_off_diagonal, params=dict(p=0.50, rescale=False)),
    'drop_p75': dict(perturb=P.dropout_off_diagonal, params=dict(p=0.75, rescale=False)),
    'drop_p90': dict(perturb=P.dropout_off_diagonal, params=dict(p=0.90, rescale=False)),
    'drop_p97': dict(perturb=P.dropout_off_diagonal, params=dict(p=0.97, rescale=False)),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lm = load_transformer_weights(family='general')
# lm_ab is only used as a not-None flag by GCNBfDataset; ProtBERT embeds every chain.
train_loader, val_loader, test_loader, dataset = make_split_loaders(
    DATA_DIR, device, lm_ab=True, lm_ag=lm)

RESULTS_CSV = os.path.join(CHECKPOINTS_DIR, 'sweep_results.csv')
RESULT_COLUMNS = ['config', 'best_epoch', 'n_replicates', 'threshold', 'prevalence', 'n',
                  'mcc', 'mcc_std', 'f1', 'precision', 'recall',
                  'auroc', 'auprc', 'auprc_std', 'auprc_over_chance',
                  'accuracy', 'tp', 'fp', 'fn', 'tn']


def append_results_row(name, metrics):
    is_new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow(RESULT_COLUMNS)
        if 'error' in metrics:
            w.writerow([name, 'FAILED'] + [''] * (len(RESULT_COLUMNS) - 2))
            return
        row = [name]
        for col in RESULT_COLUMNS[1:]:
            v = metrics.get(col, '')
            row.append(round(v, 4) if isinstance(v, float) else v)
        w.writerow(row)


def run_config(name, cfg):
    perturb, params = cfg.get('perturb'), cfg.get('params', {})
    tag = f'h{args.hidden_channels}_lr{args.lr}__{name}__s{SEED}'
    out_dir = os.path.join(CHECKPOINTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(SEED)
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.out_channels,
        lm_dim=1024,
        lm=None,                                
        seq_only=cfg.get('seq_only', False),    
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auprc, best_epoch, best_state, bad = -1.0, -1, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_perturbed(model, optimiser, train_loader, len(train_loader),
                               perturb=perturb, perturb_params=params)

        val, *_ = evaluate(model, val_loader, perturb, params)
        history.append({'epoch': epoch, 'loss': loss, 'val_auprc': val['auprc'],
                        'secs': round(time.time() - t0, 1)})
        print(f"[{name}] ep {epoch:3d}  loss {loss:.4f}  val AUPRC {val['auprc']:.4f}"
              f"  ({history[-1]['secs']}s)")

        if val['auprc'] > best_auprc:
            best_auprc, best_epoch, bad = val['auprc'], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f'[{name}] early stop at ep {epoch}; best was {best_epoch}')
                break

    if best_state is None:
        raise RuntimeError('no epoch completed')
    model.load_state_dict(best_state)

    _, vl, vy, *_ = evaluate(model, val_loader, perturb, params)
    thr = pick_threshold(vl, vy, objective='mcc')

    reps = cfg.get('eval_replicates', 1)
    runs = [evaluate(model, test_loader, perturb, params, threshold=thr, seed=k)
            for k in range(reps)]
    metrics, logits, labels, pdbs, sizes = runs[0]
    if reps > 1:
        per_rep = [r[0] for r in runs]
        metrics = {'n': per_rep[0]['n'], 'prevalence': per_rep[0]['prevalence'],
                   'threshold': thr, 'replicates': per_rep,
                   'accuracy': per_rep[0]['accuracy'],
                   'tp': per_rep[0]['tp'], 'fp': per_rep[0]['fp'],
                   'fn': per_rep[0]['fn'], 'tn': per_rep[0]['tn']}
        for key in ('mcc', 'f1', 'precision', 'recall', 'auroc', 'auprc'):
            metrics[key] = float(np.mean([m[key] for m in per_rep]))
            metrics[key + '_std'] = float(np.std([m[key] for m in per_rep]))
        metrics['auprc_over_chance'] = metrics['auprc'] / max(metrics['prevalence'], 1e-12)

    metrics['best_epoch'] = best_epoch
    metrics['n_replicates'] = reps

    torch.save(best_state, os.path.join(out_dir, 'model.pth'))
    torch.save({'logits': logits.cpu(), 'labels': labels.cpu(),   # replicate 0
                'pdb': pdbs, 'sizes': sizes}, os.path.join(out_dir, 'test_logits.pt'))
    json.dump({'name': name, 'params': params, 'best_epoch': best_epoch,
               'val_auprc': best_auprc, 'threshold': thr, 'args': vars(args)},
              open(os.path.join(out_dir, 'config.json'), 'w'), indent=2, default=str)
    json.dump({'test': metrics, 'history': history},
              open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=2)

    print(f"[{name}] TEST  MCC {metrics['mcc']:.4f}  F1 {metrics['f1']:.4f}  "
          f"P {metrics['precision']:.4f}  R {metrics['recall']:.4f}  "
          f"AUROC {metrics['auroc']:.4f}  AUPRC {metrics['auprc']:.4f} "
          f"({metrics['auprc_over_chance']:.2f}x azar)  "
          f"[thr {metrics['threshold']:.3f}, prev {metrics['prevalence']:.3f}]")
    
    return metrics


names = [args.only] if args.only else list(CONFIGS)
summary = {}
for name in names:
    try:
        summary[name] = run_config(name, CONFIGS[name])
    except Exception as exc:                    # one bad config must not kill the sweep
        print(f'[{name}] FAILED: {type(exc).__name__}: {exc}')
        summary[name] = {'error': f'{type(exc).__name__}: {exc}'}
    append_results_row(name, summary[name])
    json.dump(summary, open(os.path.join(CHECKPOINTS_DIR, 'sweep_summary.json'), 'w'),
              indent=2, default=str)