r"""BLCA 5-fold cross-validation using pre-defined fold splits.

Usage:
    python -u run_5fold.py
"""
import subprocess
import sys

CMD = [
    sys.executable, "-u", "survmi/train_csv.py",
    "--pathway-mask", "blca_pathway_mask.pt",
    "--epochs", "30",
    "--batch-size", "8",
    "--max-patches", "2048",
    "--hidden-dim", "128",
    "--lr", "5e-5",
]

val_c_list = []

for fold in range(5):
    train_csv = f"blca_fold{fold}_train.csv"
    val_csv = f"blca_fold{fold}_val.csv"
    out_dir = f"runs/blca_5fold/fold{fold}"

    args = CMD + [
        "--csv", train_csv,
        "--val-csv", val_csv,
        "--out-dir", out_dir,
    ]
    print(f"\n{'='*60}")
    print(f"FOLD {fold}/5")
    print(f"{'='*60}")
    print(" ".join(args))

    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"Fold {fold} FAILED (exit {result.returncode})")
        continue

    # 读取 best validation c-index
    import json
    try:
        with open(f"{out_dir}/config.json") as f:
            config = json.load(f)
        # config 在运行时输出 best epoch 信息；从 history 读取最优
    except FileNotFoundError:
        pass

    with open(f"{out_dir}/history.jsonl") as f:
        best_c = max(
            json.loads(line).get("val_c_index", -1.0)
            for line in f if line.strip()
        )
    val_c_list.append(best_c)
    print(f"Fold {fold}: best val_c_index = {best_c:.4f}")

print(f"\n{'='*60}")
print("5-FOLD CV RESULTS")
print(f"{'='*60}")
for i, c in enumerate(val_c_list):
    print(f"  Fold {i}: {c:.4f}")
if val_c_list:
    import statistics
    mean_c = statistics.mean(val_c_list)
    std_c = statistics.stdev(val_c_list) if len(val_c_list) > 1 else 0.0
    print(f"\n  Mean C-index: {mean_c:.4f} +/- {std_c:.4f}")
