import argparse
import csv
from pathlib import Path


def tcga_case_id(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return stem[:12]
    return "-".join(parts[:3])


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def truthy_event(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "dead", "deceased", "event", "yes"}:
        return "1"
    if text in {"0", "false", "alive", "censored", "no"}:
        return "0"
    try:
        return "1" if float(text) > 0 else "0"
    except ValueError:
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SurvMI CSV manifest from TCGA-style files.")
    parser.add_argument("--patch-dir", required=True, help="Directory containing WSI patch feature .pt/.npy files")
    parser.add_argument("--clinical-csv", required=True, help="Clinical table with case id, time, event columns")
    parser.add_argument("--omics-csv", required=True, help="Wide omics table: one row per case/sample, gene columns")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--case-col", default="case_id")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--event-col", default="event")
    parser.add_argument("--omics-case-col", default="case_id")
    parser.add_argument("--gene-prefix", default="gene:")
    parser.add_argument("--feature-ext", default=".pt")
    parser.add_argument("--gene-list-out", default=None, help="Optional one-gene-per-line file in output CSV order")
    args = parser.parse_args()

    patch_dir = Path(args.patch_dir)
    clinical_rows = read_table(Path(args.clinical_csv))
    omics_rows = read_table(Path(args.omics_csv))

    clinical = {}
    for row in clinical_rows:
        case = row[args.case_col].strip()
        if case:
            clinical[case] = row

    omics = {}
    omics_columns = None
    for row in omics_rows:
        case = row[args.omics_case_col].strip()
        if not case:
            continue
        if omics_columns is None:
            omics_columns = []
            for c in row.keys():
                if c == args.omics_case_col:
                    continue
                try:
                    float(row[c])
                except (TypeError, ValueError):
                    continue
                omics_columns.append(c)
        omics[case] = row
    omics_columns = omics_columns or []

    patch_files = sorted(p for p in patch_dir.glob(f"*{args.feature_ext}") if p.is_file())
    out_rows = []
    missing_clinical = 0
    missing_omics = 0
    for patch_path in patch_files:
        case = tcga_case_id(patch_path.name)
        if case not in clinical:
            missing_clinical += 1
            continue
        if case not in omics:
            missing_omics += 1
            continue
        c_row = clinical[case]
        o_row = omics[case]
        out = {
            "slide_id": patch_path.stem,
            "case_id": case,
            "patch_path": str(patch_path),
            "time": c_row[args.time_col],
            "event": truthy_event(c_row[args.event_col]),
        }
        for col in omics_columns:
            out[f"{args.gene_prefix}{col}"] = o_row[col]
        out_rows.append(out)

    if not out_rows:
        raise SystemExit("No matched rows. Check case columns and feature filenames.")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        f"wrote {len(out_rows)} rows to {out_path} "
        f"(missing_clinical={missing_clinical}, missing_omics={missing_omics})"
    )
    if args.gene_list_out:
        gene_path = Path(args.gene_list_out)
        gene_path.parent.mkdir(parents=True, exist_ok=True)
        gene_path.write_text("\n".join(omics_columns) + "\n", encoding="utf-8")
        print(f"wrote {len(omics_columns)} ordered genes to {gene_path}")


if __name__ == "__main__":
    main()
