from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    cox_ph_loss,
    pair_level_patch_pathway_nce_loss,
    risk_aware_info_nce_loss,
)
from .modules import (
    MIGatedCrossAttention,
    MorphologyPrototypePooler,
    PathwayProjector,
)


class CorePatchPathwaySurvMI(nn.Module):
    """唯一主线模型：survival-conditioned patch-to-pathway MI selection。

    Cox loss + risk-aware InfoNCE + sparse MI alignment。
    这是全项目唯一保留的模型（旧的 shared-specific IB / patch-pathway /
    prognostic-conflict / missing-modality 全家桶已按审稿收敛决策删除）。
    """

    def __init__(
        self,
        wsi_dim: int,
        omics_dim: int,
        hidden_dim: int = 256,
        num_prototypes: int = 32,
        num_pathways: int = 50,
        mi_gate: float = 1.0,
        pathway_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.proto_pool = MorphologyPrototypePooler(wsi_dim, hidden_dim, num_prototypes)
        if pathway_mask is not None:
            num_pathways = pathway_mask.size(0)
        self.uses_pathway_mask = pathway_mask is not None
        self.pathway_encoder = PathwayProjector(omics_dim, hidden_dim, hidden_dim, num_pathways, pathway_mask)
        self.mi_proj_p = nn.Linear(hidden_dim, hidden_dim)
        self.mi_proj_g = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = MIGatedCrossAttention(hidden_dim, num_heads=4, mi_gate=mi_gate)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 4),
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        patches: torch.Tensor,
        omics: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        proto_tokens, proto_assign = self.proto_pool(patches, patch_mask)
        pathway_tokens = self.pathway_encoder(omics)

        proto_mi = F.normalize(self.mi_proj_p(proto_tokens), dim=-1)
        pathway_mi = F.normalize(self.mi_proj_g(pathway_tokens), dim=-1)
        mi_matrix = torch.einsum("bkd,bmd->bkm", proto_mi, pathway_mi)
        mi_bias = (mi_matrix - mi_matrix.mean(dim=(1, 2), keepdim=True)) / mi_matrix.std(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-6)

        pathway_context, cross_attention = self.cross_attn(proto_tokens, pathway_tokens, mi_bias)
        proto_global = proto_tokens.mean(dim=1)
        pathway_global = pathway_tokens.mean(dim=1)
        context_global = pathway_context.mean(dim=1)
        mi_flat = mi_matrix.flatten(start_dim=1)
        mi_summary = torch.stack(
            [
                mi_flat.mean(dim=1),
                mi_flat.amax(dim=1),
                mi_flat.topk(k=min(4, mi_flat.size(1)), dim=1).values.mean(dim=1),
                mi_flat.std(dim=1),
            ],
            dim=1,
        )
        risk = self.risk_head(torch.cat([proto_global, context_global, mi_summary], dim=-1)).squeeze(-1)

        return {
            "risk": risk,
            "proto_tokens": proto_tokens,
            "pathway_tokens": pathway_tokens,
            "proto_global": proto_global,
            "pathway_global": pathway_global,
            "proto_mi_tokens": proto_mi,
            "pathway_mi_tokens": pathway_mi,
            "mi_matrix": mi_matrix,
            "mi_bias": mi_bias,
            "mi_summary": mi_summary,
            "proto_assign": proto_assign,
            "cross_attention": cross_attention,
        }

    def loss(
        self,
        output: dict[str, torch.Tensor],
        time: torch.Tensor,
        event: torch.Tensor,
        weights: Optional[dict[str, float]] = None,
        risk_cutoff: Optional[float | torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        weights = weights or {}
        surv = cox_ph_loss(output["risk"], time, event)
        pp_mi = pair_level_patch_pathway_nce_loss(
            output["proto_mi_tokens"],
            output["pathway_mi_tokens"],
            time,
            event,
            cutoff=risk_cutoff,
        )
        global_mi = risk_aware_info_nce_loss(
            output["proto_global"],
            output["pathway_global"],
            time,
            event,
            cutoff=risk_cutoff,
        )
        sparse = output["mi_matrix"].abs().mean()

        total = (
            weights.get("surv", 1.0) * surv
            + weights.get("pp_mi", 0.1) * pp_mi
            + weights.get("global_mi", 0.0) * global_mi
            + weights.get("sparse", 0.001) * sparse
        )
        return {
            "total": total,
            "surv": surv.detach(),
            "pp_mi": pp_mi.detach(),
            "global_mi": global_mi.detach(),
            "sparse": sparse.detach(),
        }
