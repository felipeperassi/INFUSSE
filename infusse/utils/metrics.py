import csv
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def epitope_metrics(logits, labels, run_name=None, epochs=None, lr=None, pos_weight=None, n_test=None):
    prob, y = torch.sigmoid(logits).numpy(), labels.numpy().astype(int)
    pred = (prob > 0.5).astype(int)
    
    tp = int((pred * y).sum()); fp = int((pred * (1 - y)).sum())
    fn = int(((1 - pred) * y).sum()); tn = int(((1 - pred) * (1 - y)).sum())
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    mcc = (tp * tn - fp * fn) / den if den else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    auroc = float(roc_auc_score(y, prob))
    auprc = float(average_precision_score(y, prob))

    print(f'Metrics: MCC={mcc:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, AUROC={auroc:.4f}, AUPRC={auprc:.4f}')
    return {'config': run_name, 'epochs': epochs, 'lr': lr,
            'pos_weight': round(pos_weight, 3) if pos_weight is not None else None,
            'n_test': n_test, 'n': len(y),
            'prevalence': round(float(y.mean()), 4),
            'mcc': round(mcc, 4),
            'f1': round(f1, 4),
            'precision': round(prec, 4), 
            'recall': round(rec, 4),
            'auroc': round(auroc, 4),
            'auprc': round(auprc, 4),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def save_metrics(row, logits, labels, run_dir, csv_path):
    torch.save({'logits': logits, 'labels': labels}, os.path.join(run_dir, 'test_logits.pt'))
    with open(csv_path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if fh.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    print(row)