import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from survmi import PCBConfig, PrognosticConflictBottleneck
from survmi.data import (
    SurvivalOmicsCSVDataset,
    compute_global_event_cutoff,
    fit_omics_normalizer,
    infer_feature_dims,
    load_pathway_mask,
    pad_collate,
    patient_level_split_indices,
)
from survmi.metrics import RunningMean, concordance_index


def build_model(wsi_dim: int, omics_dim: int, hidden_dim: int, pathway_mask=None, mode: str = "full"):
    return PrognosticConflictBottleneck(
        wsi_dim=wsi_dim,
        omics_dim=omics_dim,
        hidden_dim=hidden_dim,
        pathway_mask=pathway_mask,
        config=PCBConfig(mode=mode),
    )


def run_epoch(model, loader, device, optimizer=None, risk_cutoff=None, desc="train"):
    is_train = optimizer is not None
    model.train(is_train)
    loss_meter = RunningMean()
    risks = []
    times = []
    events = []
    slide_ids = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 5)  # print ~5 times per epoch

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for i, batch in enumerate(loader):
            patches = batch["patches"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            omics = batch["omics"].to(device)
            time = batch["time"].to(device)
            event = batch["event"].to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            output = model(patches, omics, patch_mask)
            losses = model.loss(output, time, event, risk_cutoff=risk_cutoff)

            if is_train:
                losses["total"].backward()
                optimizer.step()

            batch_size = time.numel()
            loss_meter.update(losses["total"].item(), batch_size)
            risks.append(output["risk"].detach().cpu())
            times.append(time.detach().cpu())
            events.append(event.detach().cpu())
            slide_ids.extend(batch["slide_id"])

            if i == 0 or (i + 1) % log_every == 0:
                print(f"  {desc} batch {i+1}/{n_batches} loss={losses['total'].item():.3f}", flush=True)

    risk = torch.cat(risks) if risks else torch.empty(0)
    time = torch.cat(times) if times else torch.empty(0)
    event = torch.cat(events) if events else torch.empty(0)
    c_index = concordance_index(risk, time, event) if risk.numel() else float("nan")
    return {
        "loss": loss_meter.value,
        "c_index": c_index,
        "risk": risk,
        "time": time,
        "event": event,
        "slide_id": slide_ids,
    }


def write_predictions(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "risk", "time", "event"])
        writer.writeheader()
        for slide_id, risk, time, event in zip(
            result["slide_id"],
            result["risk"].tolist(),
            result["time"].tolist(),
            result["event"].tolist(),
        ):
            writer.writerow({"slide_id": slide_id, "risk": risk, "time": time, "event": event})


def collect_cases(dataset, indices=None) -> list[str]:
    if indices is None:
        indices = range(len(dataset))
    return sorted({dataset.get_case_id(int(idx)) for idx in indices})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with patch_path, time, event, and omics_path or gene:* columns")
    parser.add_argument("--val-csv", default=None, help="Optional validation CSV. If omitted, split --csv.")
    parser.add_argument("--root", default=None, help="Root directory for relative feature paths")
    parser.add_argument("--val-root", default=None, help="Root directory for validation CSV relative feature paths")
    parser.add_argument(
        "--mode",
        choices=["full", "agreement_only", "conflict_only"],
        default="full",
        help="消融开关：full=完整 PCB；agreement_only=对齐融合 baseline；conflict_only=仅冲突通道",
    )
    parser.add_argument("--epochs", type=int, default=5)
    # Cox 偏似然的风险集只覆盖当前 batch。真实数据审查率高，batch 过小会导致
    # 每个 batch 只有 1-2 个事件、风险集估计噪声大且有偏，严重削弱学习信号。
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-patches", type=int, default=4096)
    parser.add_argument("--pathway-mask", default=None, help="Optional real pathway mask [num_pathways, num_genes]")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="runs/core_csv")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="下采样后的 patch 缓存目录。首轮加载后缓存小文件，之后每个 epoch 秒读，大幅加速。",
    )
    parser.add_argument("--no-omics-log1p", action="store_true", help="Disable log1p before train-fit z-score")
    parser.add_argument("--no-omics-zscore", action="store_true", help="Disable train-fit omics z-score")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cache = (str(Path(args.cache_dir) / "train") if args.cache_dir else None)
    val_cache = (str(Path(args.cache_dir) / "val") if args.cache_dir else None)
    dataset = SurvivalOmicsCSVDataset(
        args.csv, root=args.root, max_patches=args.max_patches, seed=args.seed, cache_dir=train_cache
    )
    pathway_mask = load_pathway_mask(args.pathway_mask) if args.pathway_mask else None
    if pathway_mask is None:
        print("warning: no real pathway mask was provided; pathway tokens are learnable linear combinations.")
    if args.val_csv:
        train_set = dataset
        val_set = SurvivalOmicsCSVDataset(
            args.val_csv,
            root=args.val_root or args.root,
            max_patches=args.max_patches,
            seed=args.seed,
            cache_dir=val_cache,
        )
        train_indices_for_stats = list(range(len(dataset)))
        split_cases = {
            "train_cases": collect_cases(dataset),
            "val_cases": collect_cases(val_set),
        }
    else:
        train_indices, val_indices = patient_level_split_indices(dataset, args.val_fraction, args.seed)
        train_indices_for_stats = train_indices
        if val_indices:
            train_set = torch.utils.data.Subset(dataset, train_indices)
            val_set = torch.utils.data.Subset(dataset, val_indices)
        else:
            train_set, val_set = dataset, None
        split_cases = {
            "train_cases": collect_cases(dataset, train_indices),
            "val_cases": collect_cases(dataset, val_indices),
        }

    if args.no_omics_zscore:
        mean, std = None, None
    else:
        mean, std = fit_omics_normalizer(dataset, train_indices_for_stats, log1p=not args.no_omics_log1p)
    dataset.set_omics_normalizer(mean, std, log1p=not args.no_omics_log1p)
    if args.val_csv and val_set is not None and hasattr(val_set, "set_omics_normalizer"):
        val_set.set_omics_normalizer(mean, std, log1p=not args.no_omics_log1p)

    wsi_dim, omics_dim = infer_feature_dims(dataset)
    if pathway_mask is not None and pathway_mask.size(1) != omics_dim:
        raise ValueError(f"pathway mask gene dimension {pathway_mask.size(1)} does not match omics dim {omics_dim}")
    risk_cutoff = compute_global_event_cutoff(dataset, train_indices_for_stats)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=args.num_workers,
    )
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=args.num_workers,
        )
    else:
        val_loader = None

    model = build_model(wsi_dim, omics_dim, args.hidden_dim, pathway_mask, mode=args.mode).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val_c = -1.0
    best_epoch = 0

    config = vars(args).copy()
    config.update(
        {
            "wsi_dim": wsi_dim,
            "omics_dim": omics_dim,
            "risk_cutoff": risk_cutoff,
            "risk_cutoff_source": "train_only",
            "split_level": "patient",
            "omics_log1p": not args.no_omics_log1p,
            "omics_zscore": not args.no_omics_zscore,
            "train_size": len(train_set),
        }
    )
    if val_set is not None:
        config["val_size"] = len(val_set)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if mean is not None and std is not None:
        torch.save({"mean": mean, "std": std, "log1p": not args.no_omics_log1p}, out_dir / "omics_normalizer.pt")
    (out_dir / "split_cases.json").write_text(json.dumps(split_cases, indent=2), encoding="utf-8")

    for epoch in range(args.epochs):
        train_result = run_epoch(
            model,
            train_loader,
            args.device,
            optimizer=optimizer,
            risk_cutoff=risk_cutoff,
            desc="train",
        )
        log = {
            "epoch": epoch + 1,
            "train_loss": train_result["loss"],
            "train_c_index": train_result["c_index"],
        }
        msg = f"epoch={epoch + 1} train_loss={train_result['loss']:.4f} train_c={train_result['c_index']:.4f}"
        if val_loader is not None:
            val_result = run_epoch(model, val_loader, args.device, risk_cutoff=risk_cutoff, desc="val")
            log.update({"val_loss": val_result["loss"], "val_c_index": val_result["c_index"]})
            msg += f" val_loss={val_result['loss']:.4f} val_c={val_result['c_index']:.4f}"
            if val_result["c_index"] == val_result["c_index"] and val_result["c_index"] > best_val_c:
                best_val_c = val_result["c_index"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": best_epoch,
                        "val_c_index": best_val_c,
                        "config": config,
                    },
                    out_dir / "best.pt",
                )
                write_predictions(out_dir / "best_val_predictions.csv", val_result)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "config": config}, out_dir / "last.pt")
        with (out_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        print(msg)

    if val_loader is not None:
        print(f"best_epoch={best_epoch} best_val_c_index={best_val_c:.4f}")


if __name__ == "__main__":
    main()
