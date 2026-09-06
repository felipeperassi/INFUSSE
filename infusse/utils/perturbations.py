import torch

def _ab_ag(C):
    C = C.long()
    return (C < 2), (C == 2) # AB H/L=0/1, AG=2

def _undirected(edge_index): # (i,j) and (j,i) share an id, so both get the same perturbation
    i, j = edge_index[0], edge_index[1]
    min, max = torch.minimum(i, j), torch.maximum(i, j)

    key = min * (int(edge_index.max()) + 1) + max
    uniq, inv = torch.unique(key, return_inverse=True)

    return uniq.shape[0], inv

# Training structure perturbations

def remove_off_diagonal(edge_index, edge_attr, C): # Remove all edges between AB and AG
    is_ab, _ = _ab_ag(C)
    keep = is_ab[edge_index[0]] == is_ab[edge_index[1]] 
    return edge_index[:, keep], edge_attr[keep]

def dropout_off_diagonal(edge_index, edge_attr, C, p, seed=0, rescale=False): # Drop edges between AB and AG with probability p, rescale remaining edges by 1/(1-p)
    if (rescale and (p >= 1.0 or p < 0.0)):
        raise ValueError("Rescaling is not possible when p >= 1 or p < 0")
    
    is_ab, _ = _ab_ag(C)
    interaction = is_ab[edge_index[0]] != is_ab[edge_index[1]]        
    n_uniq, inv = _undirected(edge_index)

    g = None if seed is None else torch.Generator(device='cpu').manual_seed(seed)
    drop_u = (torch.rand(n_uniq, generator=g) < p).to(edge_index.device)   
    drop = drop_u[inv] & interaction

    keep =  ~drop

    if rescale:
        edge_attr = edge_attr.clone()
        edge_attr[interaction] = edge_attr[interaction] * (1.0 / (1.0 - p))

    return edge_index[:, keep], edge_attr[keep]

def off_diagonal_shuffle(edge_index, edge_attr, C, seed=0): # A[i,j] -> A[i, perm(j)] for j in AG
    is_ab, is_ag = _ab_ag(C)
    interaction = is_ab[edge_index[0]] != is_ab[edge_index[1]]
    if not bool(interaction.any()):
        return edge_index, edge_attr

    ab_nodes = torch.nonzero(is_ab, as_tuple=True)[0]
    ag_nodes = torch.nonzero(is_ag, as_tuple=True)[0]
    n_ab, n_ag = ab_nodes.shape[0], ag_nodes.shape[0]

    local = torch.full((C.shape[0],), -1, dtype=torch.long, device=edge_index.device)
    local[ab_nodes] = torch.arange(n_ab, device=edge_index.device)
    local[ag_nodes] = torch.arange(n_ag, device=edge_index.device)

    e = torch.nonzero(interaction, as_tuple=True)[0]
    on_ab = is_ab[edge_index[0][e]]
    ab = torch.where(on_ab, edge_index[0][e], edge_index[1][e]) # AB endpoint of each interface edge
    ag = torch.where(on_ab, edge_index[1][e], edge_index[0][e]) # AG endpoint
    i_ab, i_ag = local[ab], local[ag]

    W = torch.zeros(n_ab, n_ag, device=edge_attr.device, dtype=edge_attr.dtype)
    W[i_ab, i_ag] = edge_attr[e] # The graph is complete, so the block is fully populated

    g = None if seed is None else torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(n_ag, generator=g).to(edge_index.device)

    edge_attr = edge_attr.clone()
    edge_attr[e] = W[i_ab, perm[i_ag]] 

    return edge_index, edge_attr

#---------------------------------------------------------------------------------------------------------

# # Inference-time perturbations for robustness testing

# def drop_edges(edge_index, edge_attr, C, p, seed=0):
#     i, j = edge_index[0], edge_index[1]
#     min, max = torch.minimum(i, j), torch.maximum(i, j)

#     key = min * (int(edge_index.max()) + 1) + max
#     uniq, inv = torch.unique(key, return_inverse=True)

#     generator = torch.Generator(device='cpu').manual_seed(seed)
#     keep_u = (torch.rand(uniq.shape[0], generator=generator) >= p).to(edge_index.device)

#     return edge_index[:, keep_u[inv]], edge_attr[keep_u[inv]]

# def weight_gn(edge_index, edge_attr, C, sigma, seed=0):
#     generator = torch.Generator(device='cpu').manual_seed(seed)
#     noise = (torch.randn(edge_attr.shape, generator=generator) * sigma).to(edge_attr.device)

#     return edge_index, (edge_attr + noise).clamp(min=0.0) # Clamp to non-negative values

# def off_diagonal_add_gn(edge_index, edge_attr, C, sigma, seed=0): # w -> w + N(0, sigma^2).
#     is_ab, _ = _ab_ag(C)
#     interaction = is_ab[edge_index[0]] != is_ab[edge_index[1]]
#     g = torch.Generator(device='cpu').manual_seed(seed)
#     noise = (torch.randn(edge_attr.shape, generator=g) * sigma).to(edge_attr.device)
#     edge_attr = edge_attr.clone()
#     edge_attr[interaction] = (edge_attr[interaction] + noise[interaction]).clamp(min=0.0)
#     return edge_index, edge_attr

# def off_diagonal_mult_gn(edge_index, edge_attr, C, sigma, seed=0): # w -> w * (1 + N(0, sigma^2))
#     is_ab, _ = _ab_ag(C)
#     interaction = is_ab[edge_index[0]] != is_ab[edge_index[1]]

#     g = torch.Generator(device='cpu').manual_seed(seed)
#     eps = (1.0 + torch.randn(edge_attr.shape, generator=g) * sigma).to(edge_attr.device)

#     edge_attr = edge_attr.clone()
#     edge_attr[interaction] = (edge_attr[interaction] * eps[interaction]).clamp(min=0.0)
#     return edge_index, edge_attr   

# def off_diagonal_distance_gn(edge_index, edge_attr, C, sigma, seed=0, scale=64.0, d_min=3.0, uniform=False): # d -> d + N(0, sigma^2) [A] on the interface only
#     is_ab, _ = _ab_ag(C)
#     interaction = is_ab[edge_index[0]] != is_ab[edge_index[1]]

#     n_uniq, inv = _undirected(edge_index)

#     g = torch.Generator(device='cpu').manual_seed(seed)
#     if uniform:
#         eps = (torch.rand(n_uniq, generator=g) * 2.0 - 1.0) * sigma
#     else:
#         eps = torch.randn(n_uniq, generator=g) * sigma
#     eps = eps.to(edge_attr.device)[inv]
#     eps = torch.where(interaction, eps, torch.zeros_like(eps))

#     d = torch.sqrt((-scale * torch.log(edge_attr.clamp(min=1e-30))).clamp(min=0.0))
#     d = (d + eps).clamp(min=d_min)

#     return edge_index, torch.exp(-d.pow(2) / scale)

# def off_diagonal_constant(edge_index, edge_attr, C, c=None): # A[i,j] -> c on the interface (I have to test it)
#     is_ab, _ = _ab_ag(C)
#     interaction = is_ab[edge_index[0]] != is_ab[edge_index[1]]

#     edge_attr = edge_attr.clone()
#     if c is None:
#         c = edge_attr[interaction].mean() 
#     edge_attr[interaction] = c

#     return edge_index, edge_attr

# @torch.no_grad()
# def test_perturb(model, test_loader, test_size, perturb_function, **params):
#     if perturb_function is None:
#         raise ValueError("Perturbation function must be provided for testing with perturbations.")
#     model.eval()
#     test_loss = 0.0
#     all_preds = []
#     all_labels = []
#     for loader in test_loader:
#         edge_index, edge_attr = loader.edge_index, loader.edge_attr
#         edge_index, edge_attr = perturb_function(edge_index, edge_attr, loader.c, **params)
#         pred = model(loader.x, loader.x_out, edge_index, edge_attr, loader.c)[0]
#         ag_mask = (loader.c.long() == 2)
#         loss = torch.nn.BCEWithLogitsLoss()(torch.squeeze(pred)[ag_mask], torch.squeeze(loader.y)[ag_mask])
#         test_loss += loader.num_graphs * loss.item() / test_size
#         all_preds.append(torch.squeeze(pred)[ag_mask])
#         all_labels.append(torch.squeeze(loader.y)[ag_mask])

#     all_preds = torch.cat(all_preds)
#     all_labels = torch.cat(all_labels)
#     pred_binary = (torch.sigmoid(all_preds) > 0.5).float()

#     tp = ((pred_binary == 1) & (all_labels == 1)).sum().float()
#     fp = ((pred_binary == 1) & (all_labels == 0)).sum().float()
#     fn = ((pred_binary == 0) & (all_labels == 1)).sum().float()
#     tn = ((pred_binary == 0) & (all_labels == 0)).sum().float()

#     accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
#     precision = tp / (tp + fp + 1e-8)
#     recall = tp / (tp + fn + 1e-8)
#     f1 = 2 * precision * recall / (precision + recall + 1e-8)
#     mcc_num = (tp * tn - fp * fn)
#     mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
#     mcc = mcc_num / mcc_den

#     print(f'  Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}')

#     return float(test_loss), float(f1)