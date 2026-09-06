import os
import numpy as np

from infusse.config import DATA_DIR, VAL_SEED

out_path = DATA_DIR + 'val_indices.npy'
if os.path.exists(out_path):
    raise SystemExit(f'{out_path} already exists. Delete it by hand if you really want a new split.')

pdb_codes = np.load(DATA_DIR + 'pdb_codes.npy')
test_idx = set(np.load(DATA_DIR + 'test_indices.npy').tolist())

train_idx = np.array([i for i in range(len(pdb_codes)) if i not in test_idx])
rng = np.random.default_rng(VAL_SEED)
val_idx = np.sort(rng.choice(train_idx, size=int(round(0.10 * len(train_idx))), replace=False))

np.save(out_path, val_idx)
print(f'total {len(pdb_codes)} | train {len(train_idx) - len(val_idx)} | val {len(val_idx)} | test {len(test_idx)}')