import argparse
import gc
import logging
import numpy as np
import os
from pathlib import Path
import time
import torch
import csv

from torch_geometric.logging import log

from infusse.config import CHECKPOINTS_DIR, DATA_DIR, DEFAULT_GRAPH, GRAPH_TYPES
from infusse.dataset.epitope_dataset import GCNBfDataset
from infusse.model.epitope_model import GCN
from infusse.utils.biology_utils import get_transformer_tokenizer
from infusse.utils.torch_utils import count_parameters, get_dataloaders, load_lstm_weights, load_transformer_weights, test, train
from infusse.utils.metrics import epitope_metrics, save_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', choices=GRAPH_TYPES, default=DEFAULT_GRAPH)
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--seq_only', action='store_true', default=False)
parser.add_argument('--identity_cutoff', type=float, default=0.9)
parser.add_argument('--split_seed', type=int, default=0)
parser.add_argument('--test_indices_file', type=str, default=None)
parser.add_argument('--train_indices_file', type=str, default=None)
parser.add_argument('--three_way', action='store_true')
parser.add_argument('--plm', choices=['protbert', 'antiberta2', 'ankh'], default='protbert')
parser.add_argument('--lightweight_checkpoint', action='store_true')
parser.add_argument('--skip_epoch_test', action='store_true')
parser.add_argument('--mmap_edges', action='store_true')
parser.add_argument('--embedding_file', type=Path, default=None)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

if args.three_way and not (args.train_indices_file and args.test_indices_file):
    parser.error('--three_way requires --train_indices_file and --test_indices_file.')

# Name of the folder where results will be saved [IMPORTANT: I change this name in each branch w/diff variants]
results_folder_name = 'results'
run_name = ''
if args.seq_only:
    run_name = 'seq_only'
run_name += f'_seed{args.seed}'

run_dir = os.path.join(CHECKPOINTS_DIR, results_folder_name, run_name)
os.makedirs(run_dir, exist_ok=True)

def save_model(model, path):
    if not args.lightweight_checkpoint:
        torch.save(model, path)
        return
    state = {name: value for name, value in model.state_dict().items() if not name.startswith('lm.')}
    torch.save({
        'state_dict': state,
        'in_channels': dataset.num_features,
        'hidden_channels': args.hidden_channels,
        'out_channels': dataset.out_channels,
        'lm_dim': lm_dim,
        'seq_only': model.seq_only,
        'plm': args.plm,
    }, path)

log_file_path = os.path.join(run_dir, 'log_.txt')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Started')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
lm = None

if args.lm == 'transformer':
    family = 'general' if args.plm == 'protbert' else 'antibody' if args.plm == 'antiberta2' else 'ankh'
    if args.embedding_file is None or not args.embedding_file.exists():
        lm = load_transformer_weights(family=family)
    input_file = 'gcn_inputs.pt' if args.plm == 'protbert' else f'gcn_inputs_{args.plm}.pt'
    special_token_ids = get_transformer_tokenizer(args.plm).all_special_ids
    train_loader, test_loader, test_size, dataset = get_dataloaders(
        DATA_DIR,
        device,
        mode='train',
        lm_ab=lm,
        lm_ag=lm,
        identity_cutoff=args.identity_cutoff,
        split_seed=args.split_seed,
        test_indices_file=args.test_indices_file,
        train_indices_file=args.train_indices_file,
        input_file=input_file,
        special_token_ids=None if args.plm == 'protbert' else special_token_ids,
        graph_type=args.graphs,
        mmap_edges=args.mmap_edges,
        embedding_file=args.embedding_file,
    )
    lm_dim = dataset.X_out[0].shape[-1]
elif args.lm == 'lstm':
    input_dim = 26  
    lm_dim = 512 
    num_layers = 2
    lm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
    lm = load_lstm_weights(lm, CHECKPOINTS_DIR+'lstm_lm.hdf5')
    train_loader, test_loader, test_size, dataset = get_dataloaders(
        DATA_DIR, device, mode='train', graph_type=args.graphs,
        mmap_edges=args.mmap_edges,
    )
else:
    print('#TODO. Implement here the no-LM option')

model_lm = lm
if args.lightweight_checkpoint and args.lm == 'transformer':
    model_lm = None
    del lm
    gc.collect()

torch.manual_seed(args.seed)

model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.out_channels,
    lm_dim=lm_dim,
    lm=model_lm,
    seq_only=args.seq_only,
).to(device)

optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

print(count_parameters(model))
print(len(dataset))
logging.info('Training is starting')

times = []
initial_epochs = 10 if args.seq_only else args.epochs

n_epitope = sum(float(torch.squeeze(b.y)[b.c.long() == 2].sum()) for b in train_loader)
n_ag_total = sum(int((b.c.long() == 2).sum()) for b in train_loader)
pos_weight = (n_ag_total - n_epitope) / n_epitope
print(f'pos_weight_epitope = {pos_weight:.3f}  ({int(n_epitope)} pos / {n_ag_total - int(n_epitope)} neg)')

with open(log_file_path, 'a') as log_file:
    for epoch in range(1, initial_epochs + 1):
        start = time.time()
        loss = train(model, optimiser, train_loader, len(train_loader.dataset), pos_weight=pos_weight)
        if args.skip_epoch_test:
            test_loss, logits, labels = np.nan, np.nan, np.nan
        else:
            test_loss, logits, labels = test(model, test_loader, test_size)
            metrics = epitope_metrics(logits, labels)
            logging.info(f'Epoch: {epoch}, Loss: {loss:.4f}, F1: {metrics["f1"]:.4f}, ' f'MCC: {metrics["mcc"]:.4f}, Test Loss: {test_loss:.4f}, 'f'Time: {time.time() - start:.2f}s')
        if args.skip_epoch_test:
            log(Epoch=epoch, Loss=loss)
        else:
            log(Epoch=epoch, Loss=loss, F1=metrics["f1"], MCC=metrics["mcc"], Test_L=test_loss)
        times.append(time.time() - start)
print(f'Median time per epoch: {np.median(times):.4f}s')

if args.seq_only: # seq_only -> 1. Train Seq apart from the model (10 epochs)
                  #             2. Train the full model (Seq + Graph) (args.epochs epochs)

    save_model(model, os.path.join(run_dir, 'model_seq_only.pth'))

    logging.info('GCN-only training is starting')

    initial_weights = {name: param.clone().detach() for name, param in model.named_parameters()}

    full_model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.out_channels,
        lm_dim=lm_dim,
        lm=model_lm,
        seq_only=False,
    ).to(device)

    full_model.load_state_dict(model.state_dict())

    for name, param in full_model.named_parameters():
        param.requires_grad = ('conv' in name) or ('sequence_linear' in name) or ('aa_linear' in name) or ('lm_linear' in name) or ('c_linear' in name) 
        logging.info(f'{name} trainable? {param.requires_grad}') # just sanity check

    optimiser_full = torch.optim.AdamW(filter(lambda p: p.requires_grad, full_model.parameters()), lr=optimiser.param_groups[-1]['lr'])

    times = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train(full_model, optimiser_full, train_loader, len(train_loader.dataset), initial_weights, pos_weight=pos_weight)
        if args.skip_epoch_test:
            test_loss, logits, labels = np.nan, np.nan, np.nan
        else:
            test_loss, logits, labels = test(full_model, test_loader, test_size)
            metrics = epitope_metrics(logits, labels)
            logging.info(f'Epoch: {epoch}, Loss: {loss:.4f}, F1: {metrics["f1"]:.4f}, ' f'MCC: {metrics["mcc"]:.4f}, Test Loss: {test_loss:.4f}, 'f'Time: {time.time() - start:.2f}s')
        if args.skip_epoch_test:
            log(Epoch=epoch, Loss=loss)
        else:
            log(Epoch=epoch, Loss=loss, F1=metrics["f1"], MCC=metrics["mcc"], Test_L=test_loss)
        times.append(time.time() - start)

    print(f'Median time per epoch: {np.median(times):.4f}s')

    model = full_model

save_model(model, os.path.join(run_dir, 'model.pth')) 

# Metrics on test set
if logits is None or labels is None:
    test_loss, logits, labels = test(model, test_loader, test_size)

row = epitope_metrics(logits, labels, run_name, args.epochs, args.lr, pos_weight, len(test_loader.dataset))
save_metrics(row, logits, labels, run_dir, os.path.join(CHECKPOINTS_DIR, results_folder_name, 'results.csv'))