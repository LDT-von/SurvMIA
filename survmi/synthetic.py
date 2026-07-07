import torch
from torch.utils.data import Dataset


class SyntheticSurvivalDataset(Dataset):
    """Small synthetic WSI-omics survival dataset for smoke tests."""

    def __init__(
        self,
        num_samples: int = 64,
        num_patches: int = 128,
        wsi_dim: int = 64,
        omics_dim: int = 128,
        seed: int = 7,
    ):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.patches = torch.randn(num_samples, num_patches, wsi_dim, generator=gen)
        self.omics = torch.randn(num_samples, omics_dim, generator=gen)

        w_signal = self.patches[:, :16].mean(dim=(1, 2))
        o_signal = self.omics[:, :16].mean(dim=1)
        risk = 0.7 * w_signal + 0.9 * o_signal + 0.2 * torch.randn(num_samples, generator=gen)
        base_time = torch.exp(-risk + 0.1 * torch.randn(num_samples, generator=gen))
        censor_time = torch.rand(num_samples, generator=gen) * 2.0
        self.time = torch.minimum(base_time, censor_time).clamp_min(0.01)
        self.event = (base_time <= censor_time).float()

    def __len__(self) -> int:
        return self.patches.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "patches": self.patches[idx],
            "omics": self.omics[idx],
            "time": self.time[idx],
            "event": self.event[idx],
        }

