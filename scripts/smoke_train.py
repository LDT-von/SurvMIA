import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from survmi import CorePatchPathwaySurvMI
from survmi.synthetic import SyntheticSurvivalDataset


def build_model(name: str, wsi_dim: int, omics_dim: int, latent_dim: int):
    if name == "core":
        return CorePatchPathwaySurvMI(wsi_dim=wsi_dim, omics_dim=omics_dim, hidden_dim=128, num_prototypes=8, num_pathways=16)
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["core"], default="core")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    wsi_dim = 64
    omics_dim = 128
    latent_dim = 64
    dataset = SyntheticSurvivalDataset(num_samples=48, num_patches=96, wsi_dim=wsi_dim, omics_dim=omics_dim)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(args.model, wsi_dim, omics_dim, latent_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    iterator = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        patches = batch["patches"].to(args.device)
        omics = batch["omics"].to(args.device)
        time = batch["time"].to(args.device)
        event = batch["event"].to(args.device)

        optimizer.zero_grad(set_to_none=True)
        output = model(patches, omics)
        losses = model.loss(output, time, event)
        losses["total"].backward()
        optimizer.step()

        metrics = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
        print(f"step={step + 1} {metrics}")


if __name__ == "__main__":
    main()
