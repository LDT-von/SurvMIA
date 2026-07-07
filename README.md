# SurvMI code scaffold

This folder contains a standalone PyTorch implementation scaffold for the
information-theoretic WSI-omics survival ideas in
`MI_Multimodal_Survival_Plans`.

The code is intentionally isolated from existing baselines so it can be tested
without changing other projects.

## What is implemented

The project keeps a single mainline model, `CorePatchPathwaySurvMI`
(survival-conditioned patch-to-pathway MI selection). The earlier
"five-module" variants (shared-specific IB, patch-pathway alignment,
prognostic-conflict bottleneck, missing-modality completion) were removed
per the reviewer-driven scope decision.

- Mainline model: `CorePatchPathwaySurvMI` (`--model core`).
- Cox proportional hazards loss.
- Survival-risk-aware InfoNCE alignment (pair-level and global).
- WSI morphology prototype pooling.
- Pathway-level omics projector.
- Synthetic-data smoke test.

## Quick smoke test

```bash
python scripts/smoke_train.py --model core --steps 5
```

The smoke test uses random synthetic WSI patch features, omics vectors, and
survival labels. It is only for checking tensor shapes, gradients, and loss
plumbing. Real TCGA data loaders should be added after confirming the target
feature layout.

## CSV data format

For real features, use `scripts/train_csv.py`. The CSV needs these columns:

```text
slide_id,patch_path,omics_path,time,event
TCGA-XX-0001,/path/to/patch_features.pt,/path/to/rna.npy,10.5,1
```

Instead of `omics_path`, you can also put gene columns directly in the CSV with
a shared prefix:

```text
slide_id,patch_path,time,event,gene:TP53,gene:EGFR,gene:MYC
```

Run:

```bash
python scripts/train_csv.py --csv train.csv --model core --pathway-mask hallmark_mask.npy --out-dir runs/blca_core
```

For the `core` model, provide a real pathway mask whenever the omics vector is
gene ordered. Pathway tokens include both patient-specific pathway activity and
a learnable pathway identity embedding. Without `--pathway-mask`, pathway tokens
are only learnable linear combinations and should be treated as a non-biological
ablation.

## Build a TCGA-style manifest

If you have a patch feature directory, a clinical table, and a wide omics table,
build the training CSV with:

```bash
python scripts/build_tcga_csv.py \
  --patch-dir E:/TCGA-huggingface/uni/pt_files \
  --clinical-csv clinical_blca.csv \
  --omics-csv rna_blca.csv \
  --case-col case_id \
  --time-col time \
  --event-col event \
  --omics-case-col case_id \
  --out-csv manifests/blca_uni_rna.csv \
  --gene-list-out manifests/blca_genes.txt
```

`train_csv.py` records `history.jsonl`, `config.json`, `best.pt`, and
`best_val_predictions.csv` in `--out-dir`.

Important defaults:

- Splits are patient-level by `case_id`, never slide-level.
- Omics are transformed with train-fit `log1p + z-score` by default.
- The risk-group cutoff is computed from the training split only.
- Build pathway masks by gene name, not by position:

```bash
python scripts/build_pathway_mask.py \
  --gmt h.all.v2025.1.Hs.symbols.gmt \
  --genes manifests/blca_genes.txt \
  --out-mask manifests/hallmark_mask.pt \
  --out-pathways manifests/hallmark_pathways.json
```
