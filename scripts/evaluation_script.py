import argparse
import csv
import gc
import json
import numpy as np
import pickle
from pathlib import Path
import time
import torch

from torch_geometric.logging import log

from infusse.config import CHECKPOINTS_DIR, DATA_DIR, DEFAULT_GRAPH, GRAPH_TYPES
from infusse.dataset.dataset import GCNBfDataset
from infusse.model.model import GCN
from infusse.utils.biology_utils import get_transformer_tokenizer
from infusse.utils.torch_utils import get_dataloaders, load_legacy_model, load_transformer_weights, test

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', choices=GRAPH_TYPES, default=DEFAULT_GRAPH)
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--hidden_channels', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--plm', choices=['protbert', 'antiberta2', 'ankh'], default='protbert')
parser.add_argument('--evaluate_test_set', action='store_true')
parser.add_argument('--dump_predictions', action='store_true')
parser.add_argument('--prediction_output', type=Path, default=None)
parser.add_argument('--result_output', type=Path, default=None)
parser.add_argument('--checkpoint', type=Path, default=None)
parser.add_argument('--data_dir', type=str, default=DATA_DIR)
parser.add_argument('--test_indices_file', type=str, default=None)
parser.add_argument('--mmap_edges', action='store_true')
parser.add_argument('--embedding_file', type=Path, default=None)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

checkpoint_path = CHECKPOINTS_DIR + f'{args.graphs}_{args.lm}_features_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}/'


def load_model(path, dataset, lm):
    try:
        saved = torch.load(path, map_location=device, weights_only=False)
    except ModuleNotFoundError:
        return load_legacy_model(path, device=device)
    if not isinstance(saved, dict) or 'state_dict' not in saved:
        return load_legacy_model(path, device=device)
    model = GCN(
        in_channels=saved['in_channels'],
        hidden_channels=saved['hidden_channels'],
        out_channels=saved['out_channels'],
        lm_dim=saved['lm_dim'],
        lm=lm,
        seq_only=saved['seq_only'],
    ).to(device)
    model.load_state_dict(saved['state_dict'], strict=False)
    return model


if args.evaluate_test_set:
    if not args.data_dir.endswith('/'):
        args.data_dir += '/'
    family = 'general' if args.plm == 'protbert' else 'antibody' if args.plm == 'antiberta2' else 'ankh'
    lm = None
    if args.embedding_file is None or not args.embedding_file.exists():
        lm = load_transformer_weights(family=family)
    input_file = 'gcn_inputs.pt' if args.plm == 'protbert' else f'gcn_inputs_{args.plm}.pt'
    special_token_ids = get_transformer_tokenizer(args.plm).all_special_ids
    _, test_loader, test_size, dataset = get_dataloaders(
        args.data_dir,
        device,
        lm_ab=lm,
        lm_ag=lm,
        test_indices_file=args.test_indices_file,
        input_file=input_file,
        special_token_ids=None if args.plm == 'protbert' else special_token_ids,
        graph_type=args.graphs,
        mmap_edges=args.mmap_edges,
        embedding_file=args.embedding_file,
    )
    model_path = args.checkpoint or Path(checkpoint_path) / f'model_{args.graphs}.pth'
    del lm
    gc.collect()
    model = load_model(model_path, dataset, None).eval()
    correlations = []
    losses = []
    predictions = []
    with torch.no_grad():
        for sample in test_loader:
            pred = model(sample.x, sample.x_out, sample.edge_index, sample.edge_attr, sample.c)[0].squeeze()
            target = sample.y.squeeze()
            correlations.append(float(torch.corrcoef(torch.stack((pred, target)))[0, 1]))
            losses.append(float(torch.mean((pred - target) ** 2)))
            if args.dump_predictions:
                predictions.append({
                    'pdb': str(sample.pdb[0]),
                    'y': target.detach().cpu().numpy(),
                    'pred': pred.detach().cpu().numpy(),
                    'chain_id': sample.c.detach().cpu().numpy(),
                })
    result = {
        'model': 'S_block' if model.seq_only else 'INFUSSE',
        'R': float(np.mean(correlations)),
        'R_sd': float(np.std(correlations, ddof=1)),
        'MSE': float(np.mean(losses)),
        'n_complexes': test_size,
        'plm': args.plm,
    }
    print(json.dumps(result))
    if args.result_output:
        args.result_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_output, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result)
            writer.writeheader()
            writer.writerow(result)
    if args.dump_predictions:
        output = args.prediction_output or Path('results') / f'predictions_{args.graphs}.pkl'
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'wb') as file:
            pickle.dump(predictions, file)
    raise SystemExit

# Getting data, model and optimiser
if args.lm == 'transformer':
    lm = load_transformer_weights(family='general')
    lm_ab = load_transformer_weights(family='antibody', cssp=False)
    train_loader, test_loader, test_size, dataset = get_dataloaders(
        checkpoint_path, device, lm_ab=lm_ab, lm_ag=lm, graph_type=args.graphs,
        mmap_edges=args.mmap_edges,
    )
else:
    train_loader, test_loader, test_size, _ = get_dataloaders(
        checkpoint_path, device, graph_type=args.graphs,
        mmap_edges=args.mmap_edges,
    )
model = torch.load(checkpoint_path+f'model_{args.graphs}.pth', map_location=device)
optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

start = time.time()
#tmp_test_acc, corr = test(model, test_loader, test_size)
print(dataset[66])
tmp_test_acc, corr = test(model, dataset[76], 1) #1mlb 66, 1mlc 76
log(Corr=corr, Test=tmp_test_acc)
eval_time = time.time() - start
print(f'Median time for evaluation: {eval_time:.4f}s')
