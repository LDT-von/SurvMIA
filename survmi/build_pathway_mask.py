import argparse
import json
from pathlib import Path

import torch


def read_gene_list(path: Path) -> list[str]:
    genes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("gene:"):
            text = text[len("gene:") :]
        genes.append(text)
    if not genes:
        raise ValueError(f"No genes found in {path}")
    return genes


def read_gmt(path: Path) -> list[tuple[str, list[str]]]:
    pathways = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g for g in parts[2:] if g]
            pathways.append((name, genes))
    if not pathways:
        raise ValueError(f"No pathway entries found in {path}")
    return pathways


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a gene-name-aligned pathway mask from a GMT file.")
    parser.add_argument("--gmt", required=True, help="MSigDB/KEGG/Reactome GMT file")
    parser.add_argument("--genes", required=True, help="Text file with one omics gene column per line, in CSV order")
    parser.add_argument("--out-mask", required=True)
    parser.add_argument("--out-pathways", required=True)
    parser.add_argument("--min-overlap", type=int, default=3)
    args = parser.parse_args()

    genes = read_gene_list(Path(args.genes))
    gene_to_idx = {gene.upper(): idx for idx, gene in enumerate(genes)}

    rows = []
    names = []
    overlaps = {}
    for name, pathway_genes in read_gmt(Path(args.gmt)):
        indices = sorted({gene_to_idx[g.upper()] for g in pathway_genes if g.upper() in gene_to_idx})
        if len(indices) < args.min_overlap:
            continue
        row = torch.zeros(len(genes), dtype=torch.float32)
        row[indices] = 1.0
        rows.append(row)
        names.append(name)
        overlaps[name] = len(indices)

    if not rows:
        raise SystemExit("No pathways passed min-overlap after gene-name alignment.")

    mask = torch.stack(rows)
    out_mask = Path(args.out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask, out_mask)

    out_pathways = Path(args.out_pathways)
    out_pathways.parent.mkdir(parents=True, exist_ok=True)
    out_pathways.write_text(
        json.dumps({"pathways": names, "genes": genes, "overlap": overlaps}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote mask {tuple(mask.shape)} to {out_mask}")
    print(f"wrote pathway metadata to {out_pathways}")


if __name__ == "__main__":
    main()
