import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from transformers import BertModel, RoFormerModel

from infusse.dataset.dataset import GCNBfDataset
from infusse.utils.biology_utils import antibody_sequence_identity, sort_keys

from infusse.config import DATA_DIR

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def max_chain_identity(X, C, idx_a, idx_b, n_chain_types=3):
    identities = []
    for k in range(n_chain_types):
        seq_a = X[idx_a][C[idx_a] == k]
        seq_b = X[idx_b][C[idx_b] == k]
        if len(seq_a) == 0 or len(seq_b) == 0:
            continue
        identities.append(antibody_sequence_identity(seq_a, seq_b))
    return max(identities) if identities else 0.0

def get_dataloaders(path, device, mode='test', train_size=0.95, lm_ab=None, lm_ag=None):
    if mode == 'test':
        shuffle = False
        batch_size = 1
    else:
        shuffle = True
        batch_size = 1
        
    edge_data = torch.load(path+'edge_data.pt')
    edge_indices = edge_data['edge_index']
    edge_attributes = edge_data['edge_attr']
    X = torch.load(path+'gcn_inputs.pt')
    Y = torch.load(path+'ab_ag_labels.pt')
    C = torch.load(path+'chain_inputs.pt')
    pdb_codes = np.load(path+'pdb_codes.npy')

    dataset = GCNBfDataset(edge_indices, edge_attributes, X, Y, device=device, pdb=pdb_codes, C=C, lm_ab=lm_ab, lm_ag=lm_ag)
    if mode == 'test':
        test_indices = np.load(path+'test_indices.npy')
    train_size = int(train_size * len(dataset))  
    test_size = len(dataset) - train_size

    if mode != 'test':
        #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_indices = list(np.arange(len(dataset)))
        test_indices = []

        random_order = train_indices.copy()
        np.random.seed(0)
        np.random.shuffle(random_order)
        print(random_order)
        for i in range(len(train_indices)):
            test_idx = random_order[i]
            if len(test_indices) >= test_size:
                np.save(path+'test_indices.npy', np.array(test_indices))
                break
            add_to_test = True
            for j in train_indices:     
                if j == test_idx:
                    continue
                #identity_ab = antibody_sequence_identity(X[test_idx][:dataset.len_ab[test_idx]], X[j][:dataset.len_ab[j]])    
                #identity_ag = antibody_sequence_identity(X[test_idx][dataset.len_ab[test_idx]:], X[j][dataset.len_ab[j]:])    
                identity = max_chain_identity(X, C, test_idx, j)

                #if identity_ab >= 0.6 or identity_ag >= 0.6:
                if identity >= 0.9:# or identity.all() <= 0.2:
                    add_to_test = False
                    break

            if add_to_test:
                print('Adding a sample to the test set.')
                test_indices.append(test_idx)
                train_indices.remove(test_idx)
    print('Created a valid split, i.e., less than 0.9 training/test sequence identity.')
    train_dataset, test_dataset = [dataset[i] for i in range(len(dataset)) if i not in test_indices], [dataset[i] for i in test_indices]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return train_loader, test_loader, len(test_loader), dataset

def load_lstm_weights(model, path):
    with h5py.File(path, 'r') as f:
        weight_map = {
            'lstm.weight_ih_l0': 'model_weights/LSTM1/LSTM1/kernel:0',
            'lstm.weight_hh_l0': 'model_weights/LSTM1/LSTM1/recurrent_kernel:0',
            'lstm.bias_ih_l0': 'model_weights/LSTM1/LSTM1/bias:0',
            'lstm.bias_hh_l0': 'model_weights/LSTM1/LSTM1/bias:0',
            'lstm.weight_ih_l1': 'model_weights/LSTM2/LSTM2/kernel:0',
            'lstm.weight_hh_l1': 'model_weights/LSTM2/LSTM2/recurrent_kernel:0',
            'lstm.bias_ih_l1': 'model_weights/LSTM2/LSTM2/bias:0',
            'lstm.bias_hh_l1': 'model_weights/LSTM2/LSTM2/bias:0',
        }

        for name, param in model.named_parameters():
            if name in weight_map:
                weight_path = weight_map[name]
                weight_data = np.array(f[weight_path])

                if 'weight_ih' in name or 'weight_hh' in name:
                    param.data = torch.from_numpy(weight_data.T)
                elif 'bias' in name:
                    if '_ih' in name:
                        param.data = torch.from_numpy(weight_data[:len(weight_data)//2])
                    else:
                        param.data = torch.from_numpy(weight_data[len(weight_data)//2:])
                param.requires_grad = False # Freezing parameters

    model.eval()

    return model

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_transformer_weights(family='antibody', cssp=False):
    if family == 'antibody':
        if cssp:
            model = RoFormerModel.from_pretrained('alchemab/antiberta2-cssp')
        else:
            model = RoFormerModel.from_pretrained('alchemab/antiberta2')
    else:
        model = BertModel.from_pretrained('Rostlab/prot_bert')

    for name, param in model.named_parameters():
        param.requires_grad = False # Freezing parameters
    model.eval()

    return model

@torch.no_grad()
def plot_performance(model, loader, ca_index, cdr_positions, glob=False, res_dict=None, h_l=None, l_l=None, last=False):
    pred, pred_struct = model(loader.x, loader.x_out, loader.edge_index, loader.edge_attr, loader.c)
    pred = pred[ca_index]
    pred_struct = pred_struct[ca_index]
    y = loader.y[ca_index]
    l = h_l + l_l
    list_of_errors = []

    if not glob:
        plt.plot(np.arange(len(torch.squeeze(y).numpy())), ((torch.squeeze(pred)-torch.squeeze(y))**2).numpy())
        for i in range(len(cdr_positions)//2):
            plt.axvspan(cdr_positions[2*i], cdr_positions[2*i+1], alpha=0.1, color='green')
        plt.show()
    else:
        for i, item in enumerate(l):
        #for i, item in enumerate(l+[str(i) for i in range(len(y)-len(l))]):
            if i < len(h_l):
                item += 'H'
            else:
                item += 'L'
            #elif i < len(l_l) and i >= len(h_l):
            #    item += 'L'
            #else:
            #    item += 'AG'
            if item in res_dict:
                res_dict[item]['tot_error'] += ((torch.squeeze(pred[i])-torch.squeeze(y[i]))**2).detach().cpu().numpy()
                res_dict[item]['count'] += 1
                res_dict[item]['struct_output'] += np.abs(torch.squeeze(pred_struct[i]).detach().cpu().numpy())
            else:
                res_dict[item] = {'tot_error': ((torch.squeeze(pred[i])-torch.squeeze(y[i]))**2).detach().cpu().numpy(), 'struct_output': np.abs(torch.squeeze(pred_struct[i]).detach().cpu().numpy()), 'count': 1}
            list_of_errors.append(((torch.squeeze(pred[i])-torch.squeeze(y[i]))**2).detach().cpu().numpy())
        if last: # last PDB
            print('Placeholder. Then uncomment everything after this')
            residue_ids = sort_keys(list(res_dict.keys()))
            cdr_positions = [residue_ids.index(el) for el in ['26H', '32H', '52H', '56H', '95H', '102H']] + [residue_ids.index(el) for el in ['24L', '34L', '50L', '56L', '89L', '97L']]
            tot_error = [res_dict[id_]['tot_error'] / res_dict[id_]['count'] for id_ in residue_ids]
            struct_output = [res_dict[id_]['struct_output'] / res_dict[id_]['count'] for id_ in residue_ids]
            #plt.plot(range(len(residue_ids)), tot_error, marker='o', linestyle='-')
            #for i in range(len(cdr_positions)//2):
            #    plt.axvspan(cdr_positions[2*i], cdr_positions[2*i+1], alpha=0.1, color='green')
            print(residue_ids)
            print(tot_error)
            #plt.xlabel('Residue index')
            #plt.ylabel('MSE')
            #plt.show()
    return res_dict, list_of_errors, y, pred


# @torch.no_grad()
# def test(model, test_loader, test_size):
#     model.eval()
#     test_loss = 0.0
#     corr = 0.0
#     for loader in test_loader:
#         pred = model(loader.x, loader.x_out, loader.edge_index, loader.edge_attr, loader.c)[0]#, loader.len_ab, loader.len_ag)
#         #print(pred)
#         #print(loader.y)
#         loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(pred), torch.squeeze(loader.y))
#         test_loss += loader.num_graphs * loss.item() / test_size
#         print(loader.pdb)
#         print(loader.num_graphs * torch.corrcoef(torch.stack((torch.squeeze(pred), torch.squeeze(loader.y))))[0,1])
#         corr += loader.num_graphs * torch.corrcoef(torch.stack((torch.squeeze(pred), torch.squeeze(loader.y))))[0,1] / test_size 

#     return float(test_loss), float(corr)

@torch.no_grad()
def test(model, test_loader, test_size):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    for loader in test_loader:
        pred = model(loader.x, loader.x_out, loader.edge_index, loader.edge_attr, loader.c)[0]
        ag_mask = (loader.c.long() == 2)
        loss = torch.nn.BCEWithLogitsLoss()(torch.squeeze(pred)[ag_mask], torch.squeeze(loader.y)[ag_mask])
        test_loss += loader.num_graphs * loss.item() / test_size
        all_preds.append(torch.squeeze(pred)[ag_mask])
        all_labels.append(torch.squeeze(loader.y)[ag_mask])

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    pred_binary = (torch.sigmoid(all_preds) > 0.5).float()

    tp = ((pred_binary == 1) & (all_labels == 1)).sum().float()
    fp = ((pred_binary == 1) & (all_labels == 0)).sum().float()
    fn = ((pred_binary == 0) & (all_labels == 1)).sum().float()
    tn = ((pred_binary == 0) & (all_labels == 0)).sum().float()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mcc_num = (tp * tn - fp * fn)
    mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc = mcc_num / mcc_den

    print(f'  Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}')

    return float(test_loss), float(f1)

def train(model, optimiser, train_loader, train_size, initial_weights=None):
    model.train()
    tr_loss = 0.0
    for loader in train_loader:
        optimiser.zero_grad()
        out, struct_out = model(loader.x, loader.x_out, loader.edge_index, loader.edge_attr, loader.c)#, loader.len_ab, loader.len_ag)
        ag_mask = (loader.c.long() == 2)
        penalty_loss = 0.0
        if initial_weights:
            for name, param in model.named_parameters():
                if param.requires_grad and 'sequence_linear' in name:
                    penalty_loss += torch.sum((param - initial_weights[name]) ** 2)
        # loss = torch.nn.MSELoss(reduction='mean')(torch.squeeze(out), torch.squeeze(loader.y)) #+ 0.01 * penalty_loss #+ 0.01 * torch.sum(struct_out ** 2)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=out.device))(torch.squeeze(out)[ag_mask], torch.squeeze(loader.y)[ag_mask])
        tr_loss += loader.num_graphs * loss.item() / train_size 
        loss.backward()
        optimiser.step()
    return float(tr_loss)
