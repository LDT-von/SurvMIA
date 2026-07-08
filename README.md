# PPCB — Prognostic Conflict Bottleneck for WSI-omics Survival Analysis

BLCA 5-fold CV 实验记录。结论：**这代码在 BLCA 上完全不好使，c-index 跟抛硬币差不多。**

## BLCA 实验结果

| Fold | Val C-index |
|------|------------|
| 0 | 0.57 |
| 1 | 0.54 |
| 2 | 0.53 |
| 3 | 0.55 |
| 4 | 0.56 |
| **Mean ± Std** | **0.55 ± 0.02** |

- 基线（随机猜测）C-index = 0.50
- 训练集 C-index 轻松上 0.93，验证集惨不忍睹 → 严重过拟合
- 362 样本 × 17025 基因 × ~1M 参数，数据量根本撑不住

## 数据

| 数据源 | 路径 |
|--------|------|
| 基因表达 | `E:\TCGA_Gene_Data\drive-download-20260702T030012Z-3-001\blca_rna_inter.csv` |
| 5折划分 | `E:\TCGA-5fold\dataset_csv\splits\5fold\blca\fold_{0-4}.csv` |
| WSI 特征 | `E:\TCGA-data\CPathPatchFeature\blca\uni\pt_files\*.pt` |

## 快速上手

```powershell
# 1. 生成 BLCA manifest + pathway mask（一次性）
python -c "... 见 blca_manifest.csv 和 blca_pathway_mask.pt"

# 2. 生成 5-fold CSV
python -c "... 见 blca_fold*_train.csv 和 blca_fold*_val.csv"

# 3. 训练（单折）
python -u survmi/train_csv.py `
    --csv blca_fold0_train.csv --val-csv blca_fold0_val.csv `
    --pathway-mask blca_pathway_mask.pt `
    --epochs 30 --batch-size 8 --max-patches 2048 --hidden-dim 128 `
    --out-dir runs/blca_fold0

# 4. 5-fold CV（自动跑5折并汇总）
python -u run_5fold.py
```

## 目录结构

```
survmi/
  models.py      # PrognosticConflictBottleneck 主模型
  modules.py     # MorphologyPrototypePooler, PathwayProjector, MIGatedCrossAttention
  losses.py      # Cox, InfoNCE, orthogonality, pairwise rank
  metrics.py     # concordance_index
  data.py        # SurvivalOmicsCSVDataset
  train_csv.py   # 训练入口
  smoke_train.py # 合成数据冒烟测试
run_5fold.py     # 5-fold CV 运行脚本
test_pipeline.py # 端到端 pipeline 测试
```
