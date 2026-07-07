"""Quick pipeline test"""
import sys, time
sys.path.insert(0, '.')
from pathlib import Path
from survmi.data import (
    SurvivalOmicsCSVDataset, fit_omics_normalizer, infer_feature_dims,
    load_pathway_mask, pad_collate, patient_level_split_indices,
)
from torch.utils.data import DataLoader, Subset
from survmi import PCBConfig, PrognosticConflictBottleneck

t0 = time.time()
print('[1] Loading dataset...')
ds = SurvivalOmicsCSVDataset('blca_manifest.csv', max_patches=512)
print(f'    {len(ds)} rows in {time.time()-t0:.0f}s')

print('[2] Fitting normalizer...')
idx, vidx = patient_level_split_indices(ds, val_fraction=0.2, seed=7)
mean, std = fit_omics_normalizer(ds, idx, log1p=True)
ds.set_omics_normalizer(mean, std, log1p=True)
print(f'    omics dim: {std.shape[0]}')

print('[3] Getting dims...')
wsi_dim, omics_dim = infer_feature_dims(ds)
print(f'    wsi_dim={wsi_dim}, omics_dim={omics_dim}')

print('[4] Loading pathway mask...')
mask = load_pathway_mask('blca_pathway_mask.pt')
print(f'    {mask.shape}')

print('[5] Building model...')
model = PrognosticConflictBottleneck(wsi_dim, omics_dim, hidden_dim=64, pathway_mask=mask)
print(f'    params: {sum(p.numel() for p in model.parameters()):,}')

print('[6] Creating DataLoader...')
train_set = Subset(ds, idx[:8])
loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=pad_collate)
print(f'    train_size={len(train_set)}')

print('[7] Running one step...')
batch = next(iter(loader))
print(f'    patches shape: {batch["patches"].shape}, omics shape: {batch["omics"].shape}')
output = model(batch['patches'], batch['omics'], batch['patch_mask'])
losses = model.loss(output, batch['time'], batch['event'])
print(f'    loss={losses["total"].item():.4f}')
print('Pipeline OK!')
