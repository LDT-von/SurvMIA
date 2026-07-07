import csv
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_path(path: str, root: Optional[Path]) -> Path:
    p = Path(path)
    if not p.is_absolute() and p.exists():
        return p
    if not p.is_absolute() and root is not None:
        p = root / p
    return p


def load_tensor_file(path: Path) -> torch.Tensor:
    """Load .pt, .pth, .npy, or numeric .csv/.txt arrays."""

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.float()
        if isinstance(obj, dict):
            for key in ("features", "feat", "patch", "patches", "omics", "rna", "x"):
                if key in obj:
                    return torch.as_tensor(obj[key]).float()
        raise ValueError(f"Unsupported tensor dict keys in {path}")
    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if suffix in {".csv", ".txt"}:
        return torch.from_numpy(np.loadtxt(path, delimiter=",")).float()
    raise ValueError(f"Unsupported feature file suffix: {path.suffix}")


class SurvivalOmicsCSVDataset(Dataset):
    """CSV-backed dataset for WSI patch features plus omics vectors.

    Required columns:
        patch_path, time, event

    Omics can be provided either by:
        1. an `omics_path` column pointing to .pt/.npy/.csv vectors, or
        2. numeric columns with a shared prefix, e.g. gene:TP53.
    """

    def __init__(
        self,
        csv_path: str | Path,
        root: Optional[str | Path] = None,
        omics_prefix: str = "gene:",
        max_patches: Optional[int] = None,
        seed: int = 7,
        cache_dir: Optional[str | Path] = None,
    ):
        self.csv_path = Path(csv_path)
        self.root = Path(root) if root is not None else self.csv_path.parent
        self.omics_prefix = omics_prefix
        self.max_patches = max_patches
        self.generator = torch.Generator().manual_seed(seed)
        # 磁盘缓存：第一轮把下采样后的小 patch 张量缓存成小文件，
        # 之后每个 epoch 直接读小文件，避免重复 torch.load 整个 ~100MB 大文件。
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.omics_log1p = False
        self.omics_mean: Optional[torch.Tensor] = None
        self.omics_std: Optional[torch.Tensor] = None

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
            self.fieldnames = reader.fieldnames or []

        self.omics_columns = [name for name in self.fieldnames if name.startswith(omics_prefix)]
        if not self.rows:
            raise ValueError(f"No rows found in {self.csv_path}")
        if "patch_path" not in self.fieldnames:
            raise ValueError("CSV must include a patch_path column")
        if "time" not in self.fieldnames or "event" not in self.fieldnames:
            raise ValueError("CSV must include time and event columns")
        if "omics_path" not in self.fieldnames and not self.omics_columns:
            raise ValueError("CSV must include omics_path or numeric omics columns")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_omics_raw(self, row: dict[str, str]) -> torch.Tensor:
        if row.get("omics_path"):
            return load_tensor_file(_resolve_path(row["omics_path"], self.root)).flatten()
        values = [float(row[col]) for col in self.omics_columns]
        return torch.tensor(values, dtype=torch.float32)

    def _load_omics(self, row: dict[str, str]) -> torch.Tensor:
        omics = self._load_omics_raw(row)
        if self.omics_log1p:
            omics = torch.log1p(omics.clamp_min(0.0))
        if self.omics_mean is not None and self.omics_std is not None:
            omics = (omics - self.omics_mean) / self.omics_std.clamp_min(1e-6)
        return omics

    def set_omics_normalizer(
        self,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
        log1p: bool,
    ) -> None:
        self.omics_log1p = log1p
        self.omics_mean = None if mean is None else mean.float()
        self.omics_std = None if std is None else std.float()

    def get_case_id(self, idx: int) -> str:
        row = self.rows[idx]
        if row.get("case_id"):
            return row["case_id"]
        if row.get("slide_id"):
            return str(row["slide_id"])[:12]
        return Path(row["patch_path"]).stem[:12]

    def _cache_path(self, idx: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        row = self.rows[idx]
        stem = Path(row["patch_path"]).stem
        return self.cache_dir / f"{idx}_{stem}_mp{self.max_patches}.pt"

    def _load_patches(self, idx: int) -> torch.Tensor:
        row = self.rows[idx]
        cache_path = self._cache_path(idx)
        if cache_path is not None and cache_path.exists():
            try:
                return torch.load(cache_path, map_location="cpu")
            except Exception:
                pass  # 缓存损坏则回退到原始加载

        patches = load_tensor_file(_resolve_path(row["patch_path"], self.root))
        if patches.dim() != 2:
            raise ValueError(f"Expected patch features [num_patches, dim], got {patches.shape}")
        if self.max_patches is not None and patches.size(0) > self.max_patches:
            index = torch.randperm(patches.size(0), generator=self.generator)[: self.max_patches]
            patches = patches[index]
        patches = patches.float().contiguous()

        if cache_path is not None:
            try:
                torch.save(patches, cache_path)
            except Exception:
                pass  # 缓存写失败不影响训练
        return patches

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        patches = self._load_patches(idx)

        return {
            "patches": patches.float(),
            "omics": self._load_omics(row).float(),
            "time": torch.tensor(float(row["time"]), dtype=torch.float32),
            "event": torch.tensor(float(row["event"]), dtype=torch.float32),
            "slide_id": row.get("slide_id", Path(row["patch_path"]).stem),
            "case_id": row.get("case_id", self.get_case_id(idx)),
        }


def pad_collate(batch: Iterable[dict]) -> dict[str, torch.Tensor | list[str]]:
    batch = list(batch)
    max_patches = max(item["patches"].size(0) for item in batch)
    feat_dim = batch[0]["patches"].size(1)
    bsz = len(batch)

    patches = torch.zeros(bsz, max_patches, feat_dim, dtype=torch.float32)
    patch_mask = torch.zeros(bsz, max_patches, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["patches"].size(0)
        patches[i, :n] = item["patches"]
        patch_mask[i, :n] = True

    return {
        "patches": patches,
        "patch_mask": patch_mask,
        "omics": torch.stack([item["omics"] for item in batch]),
        "time": torch.stack([item["time"] for item in batch]),
        "event": torch.stack([item["event"] for item in batch]),
        "slide_id": [item["slide_id"] for item in batch],
        "case_id": [item["case_id"] for item in batch],
    }


def infer_feature_dims(dataset: SurvivalOmicsCSVDataset) -> tuple[int, int]:
    item = dataset[0]
    return item["patches"].size(1), item["omics"].numel()


def compute_global_event_cutoff(dataset: SurvivalOmicsCSVDataset, indices: Optional[Iterable[int]] = None) -> float:
    rows = dataset.rows if indices is None else [dataset.rows[i] for i in indices]
    times = []
    for row in rows:
        if float(row["event"]) > 0:
            times.append(float(row["time"]))
    if not times:
        times = [float(row["time"]) for row in rows]
    return float(np.median(np.asarray(times, dtype=np.float32)))


def load_pathway_mask(path: str | Path) -> torch.Tensor:
    mask = load_tensor_file(Path(path))
    if mask.dim() != 2:
        raise ValueError(f"Pathway mask must be [num_pathways, num_genes], got {mask.shape}")
    return (mask > 0).float()


def patient_level_split_indices(
    dataset: SurvivalOmicsCSVDataset,
    val_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[list[int], list[int]]:
    case_to_indices: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
        case_to_indices.setdefault(dataset.get_case_id(idx), []).append(idx)

    cases = np.asarray(sorted(case_to_indices.keys()), dtype=object)
    rng = np.random.RandomState(seed)
    rng.shuffle(cases)
    if len(cases) <= 1 or val_fraction <= 0:
        return list(range(len(dataset))), []
    val_count = int(round(len(cases) * val_fraction))
    val_count = min(max(val_count, 1), len(cases) - 1)
    val_cases = set(cases[:val_count].tolist())

    train_indices: list[int] = []
    val_indices: list[int] = []
    for case, indices in case_to_indices.items():
        if case in val_cases:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    return train_indices, val_indices


def fit_omics_normalizer(
    dataset: SurvivalOmicsCSVDataset,
    indices: Iterable[int],
    log1p: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = []
    for idx in indices:
        omics = dataset._load_omics_raw(dataset.rows[idx])
        if log1p:
            omics = torch.log1p(omics.clamp_min(0.0))
        vectors.append(omics)
    if not vectors:
        raise ValueError("Cannot fit omics normalizer on an empty split")
    matrix = torch.stack(vectors)
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std
