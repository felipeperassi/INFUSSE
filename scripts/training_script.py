import argparse
import logging
import numpy as np
import os
import time
import torch

from torch_geometric.logging import log

from infusse.config import CHECKPOINTS_DIR, DATA_DIR
from infusse.dataset.dataset import GCNBfDataset
from infusse.model.model import GCN
from infusse.utils.torch_utils import count_parameters, get_dataloaders, load_lstm_weights, load_transformer_weights, test, train

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', type=str, default='gnm')
parser.add_argument('--lm', type=str, default='transformer')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--seq_only', type=bool, default=False)
args = parser.parse_args()

log_file_path = os.path.join('..', 'log_.txt')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Started')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
lm = None

if args.lm == 'transformer':
    lm_dim = 1024
    lm_ab = load_transformer_weights(family='antibody', cssp=False)
    lm = load_transformer_weights(family='general')
    train_loader, test_loader, test_size, dataset = get_dataloaders(DATA_DIR, device, mode='train', lm_ab=lm_ab, lm_ag=lm)
elif args.lm == 'lstm':
    input_dim = 26  
    lm_dim = 512 
    num_layers = 2
    lm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
    lm = load_lstm_weights(lm, CHECKPOINTS_DIR+'lstm_lm.hdf5')
    train_loader, test_loader, test_size, dataset = get_dataloaders(DATA_DIR, device, mode='train')
else:
    print('#TODO. Implement here the no-LM option')

model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.out_channels,
    lm_dim=lm_dim,
    lm=lm, 
    seq_only=args.seq_only,
).to(device)

optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

print(count_parameters(model))
print(len(dataset))
logging.info('Training is starting')

best_val_acc = test_acc = 0
times = []
with open(log_file_path, 'a') as log_file:
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train(model, optimiser, train_loader, len(dataset)-test_size)
        tmp_test_acc, corr = test(model, test_loader, test_size)
        log_file.write(f'Epoch: {epoch}, Loss: {loss:.4f}, Corr: {corr:.4f}, Test Accuracy: {tmp_test_acc:.4f}, Time: {time.time() - start:.2f}s\n')
        log(Epoch=epoch, Loss=loss, Corr=corr, Test=tmp_test_acc)
        times.append(time.time() - start)
print(f'Median time per epoch: {np.median(times):.4f}s')

if args.seq_only:

    torch.save(model.state_dict(), CHECKPOINTS_DIR+f'model_{args.graphs}_{args.lm}_features_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}_sequence_only_.pth')

    logging.info('GCN-only training is starting')

    initial_weights = {name: param.clone().detach() for name, param in model.named_parameters()}

    full_model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.out_channels,
        lm_dim=lm_dim,
        lm=lm, 
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
        loss = train(full_model, optimiser_full, train_loader, len(dataset) - test_size, initial_weights)
        tmp_test_acc, f1 = test(full_model, test_loader, test_size)
        logging.info(f'Epoch: {epoch}, Loss: {loss:.4f}, F1: {f1:.4f}, Test Accuracy: {tmp_test_acc:.4f}, Time: {time.time() - start:.2f}s')
        log(Epoch=epoch, Loss=loss, F1=f1, Test=tmp_test_acc)
        times.append(time.time() - start)

    print(f'Median time per epoch: {np.median(times):.4f}s')

    model = full_model

torch.save(model.state_dict(), CHECKPOINTS_DIR+f'model_{args.graphs}_{args.lm}_features_hidden_channels_{args.hidden_channels}_lr_{args.lr}_epochs_{args.epochs}_sequential_{args.seq_only}_.pth')
