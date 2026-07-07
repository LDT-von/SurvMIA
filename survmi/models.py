from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    cox_ph_loss,
    orthogonality_loss,
    pairwise_rank_loss,
    risk_aware_info_nce_loss,
)
from .modules import (
    MIGatedCrossAttention,
    MorphologyPrototypePooler,
    PathwayProjector,
)


@dataclass
class PCBConfig:
    """消融开关，用于论文里证明 conflict 通道带来 agreement 拿不到的独立增益。

    mode:
      - "full"           : agreement 残差参照 + conflict 核心（主线，完整模型）
      - "agreement_only" : 关闭 conflict，退化为跨模态对齐融合（对齐类 baseline）
      - "conflict_only"  : 关闭 agreement 参照与正交约束，只留 conflict
    use_orthogonality:
        是否强制 conflict ⟂ agreement。这是"conflict = agreement 残差"的关键机制，
        关掉它可做"无残差约束"的消融。
    """

    mode: str = "full"
    use_orthogonality: bool = True

    def __post_init__(self):
        assert self.mode in {"full", "agreement_only", "conflict_only"}


class PrognosticConflictBottleneck(nn.Module):
    """主线模型：Survival-Conditioned Prognostic Conflict between Morphology and Pathway。

    创新点（novelty，写作用）：把 WSI morphology prototype 与 biological pathway token
    之间的跨模态关系，在生存条件下分解为两个互补通道——

      1. agreement（参照）：两模态*一致*的预后信息。由 risk-aware InfoNCE 对齐定义，
         并经 MI-gated cross-attention 产生一致性上下文表征。
      2. conflict（核心）：agreement *无法解释*、且与生存相关的*残差分歧*。
         通过正交约束（conflict ⟂ agreement）把 conflict 显式定义为 agreement 的残差；
         用成对排序损失把 conflict 分数对齐到两个模态风险头的分歧 |wsi_risk - omics_risk|；
         再把 conflict 摘要直接送入融合风险头，让"预后冲突"本身贡献 hazard。

    与已有工作的边界（避免 novelty 越界）：
      - 不是又一个跨模态对齐/融合方法——agreement 只是"参照系"，不是卖点；
      - 不是表征层 shared/specific 解耦（DIMAF/PIBD 等）——我们建模的是*风险层的预后
        分歧*，并通过消融证明其独立预后价值。
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
        config: Optional[PCBConfig] = None,
    ):
        super().__init__()
        self.config = config or PCBConfig()

        # 共享编码骨架
        self.proto_pool = MorphologyPrototypePooler(wsi_dim, hidden_dim, num_prototypes)
        if pathway_mask is not None:
            num_pathways = pathway_mask.size(0)
        self.uses_pathway_mask = pathway_mask is not None
        self.pathway_encoder = PathwayProjector(omics_dim, hidden_dim, hidden_dim, num_pathways, pathway_mask)

        # agreement 子空间（参照）
        self.agree_p = nn.Linear(hidden_dim, hidden_dim)
        self.agree_g = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = MIGatedCrossAttention(hidden_dim, num_heads=4, mi_gate=mi_gate)

        # conflict 子空间（核心，残差）
        self.conflict_p = nn.Linear(hidden_dim, hidden_dim)
        self.conflict_g = nn.Linear(hidden_dim, hidden_dim)

        # 单模态风险头：让"跨模态预后分歧"有明确定义
        self.wsi_risk_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.omics_risk_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        # 融合风险头：[proto_global, pathway_global, agreement_context, conflict_summary(6)]
        fused_in = hidden_dim * 3 + 6
        self.fused_risk_head = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _standardize(matrix: torch.Tensor) -> torch.Tensor:
        return (matrix - matrix.mean(dim=(1, 2), keepdim=True)) / matrix.std(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-6)

    def forward(
        self,
        patches: torch.Tensor,
        omics: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        proto_tokens, proto_assign = self.proto_pool(patches, patch_mask)   # [B, K, H]
        pathway_tokens = self.pathway_encoder(omics)                        # [B, M, H]

        # agreement 通道
        agree_p = F.normalize(self.agree_p(proto_tokens), dim=-1)
        agree_g = F.normalize(self.agree_g(pathway_tokens), dim=-1)
        agreement_matrix = torch.einsum("bkd,bmd->bkm", agree_p, agree_g)   # [B, K, M]
        agreement_bias = self._standardize(agreement_matrix)
        agreement_context, cross_attention = self.cross_attn(proto_tokens, pathway_tokens, agreement_bias)

        # conflict 通道（残差）
        conflict_p = F.normalize(self.conflict_p(proto_tokens), dim=-1)
        conflict_g = F.normalize(self.conflict_g(pathway_tokens), dim=-1)
        conflict_matrix = torch.einsum("bkd,bmd->bkm", conflict_p, conflict_g)  # [B, K, M]
        conflict_activation = torch.sigmoid(conflict_matrix)

        proto_global = proto_tokens.mean(dim=1)
        pathway_global = pathway_tokens.mean(dim=1)
        context_global = agreement_context.mean(dim=1)

        wsi_risk = self.wsi_risk_head(proto_global).squeeze(-1)
        omics_risk = self.omics_risk_head(pathway_global).squeeze(-1)
        risk_gap = (wsi_risk - omics_risk).abs()

        flat_conflict = conflict_activation.flatten(start_dim=1)
        topk = min(4, flat_conflict.size(1))
        conflict_score = flat_conflict.topk(k=topk, dim=1).values.mean(dim=1)
        conflict_summary = torch.stack(
            [
                flat_conflict.mean(dim=1),
                flat_conflict.amax(dim=1),
                conflict_score,
                flat_conflict.std(dim=1),
                wsi_risk,
                omics_risk,
            ],
            dim=1,
        )  # [B, 6]

        # 消融：关闭对应通道对融合风险头的贡献
        mode = self.config.mode
        if mode == "agreement_only":
            conflict_summary = torch.zeros_like(conflict_summary)
        elif mode == "conflict_only":
            context_global = torch.zeros_like(context_global)

        fused = torch.cat([proto_global, pathway_global, context_global, conflict_summary], dim=-1)
        risk = self.fused_risk_head(fused).squeeze(-1)

        return {
            "risk": risk,
            "wsi_risk": wsi_risk,
            "omics_risk": omics_risk,
            "risk_gap": risk_gap,
            "proto_tokens": proto_tokens,
            "pathway_tokens": pathway_tokens,
            "agree_proto": agree_p,
            "agree_pathway": agree_g,
            "conflict_proto": conflict_p,
            "conflict_pathway": conflict_g,
            "agreement_matrix": agreement_matrix,
            "conflict_matrix": conflict_matrix,
            "conflict_activation": conflict_activation,
            "conflict_score": conflict_score,
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
        cfg = self.config
        zero = output["risk"].sum() * 0.0

        compute_agreement = cfg.mode in {"full", "agreement_only"}
        compute_conflict = cfg.mode in {"full", "conflict_only"}
        compute_orth = cfg.use_orthogonality and cfg.mode == "full"

        # 主任务：融合风险的 Cox
        surv = cox_ph_loss(output["risk"], time, event)

        # 单模态风险 Cox：校准两个模态各自的风险视角，使"分歧"有意义
        wsi_surv = cox_ph_loss(output["wsi_risk"], time, event)
        omics_surv = cox_ph_loss(output["omics_risk"], time, event)

        # agreement：risk-aware InfoNCE 对齐 agreement 子空间（定义"一致"）
        if compute_agreement:
            agreement_mi = risk_aware_info_nce_loss(
                output["agree_proto"].mean(dim=1),
                output["agree_pathway"].mean(dim=1),
                time,
                event,
                cutoff=risk_cutoff,
            )
        else:
            agreement_mi = zero

        # conflict = agreement 的残差：正交约束
        if compute_orth:
            orth = 0.5 * (
                orthogonality_loss(output["agree_proto"], output["conflict_proto"])
                + orthogonality_loss(output["agree_pathway"], output["conflict_pathway"])
            )
        else:
            orth = zero

        # conflict 分数对齐模态风险分歧 + 稀疏
        if compute_conflict:
            conflict_rank = pairwise_rank_loss(output["conflict_score"], output["risk_gap"].detach())
            conflict_sparse = output["conflict_activation"].mean()
        else:
            conflict_rank = zero
            conflict_sparse = zero

        total = (
            weights.get("surv", 1.0) * surv
            + weights.get("wsi_surv", 0.2) * wsi_surv
            + weights.get("omics_surv", 0.2) * omics_surv
            + weights.get("agreement_mi", 0.1) * agreement_mi
            + weights.get("orth", 0.05) * orth
            + weights.get("conflict_rank", 0.1) * conflict_rank
            + weights.get("conflict_sparse", 0.001) * conflict_sparse
        )
        return {
            "total": total,
            "surv": surv.detach(),
            "wsi_surv": wsi_surv.detach(),
            "omics_surv": omics_surv.detach(),
            "agreement_mi": agreement_mi.detach(),
            "orth": orth.detach(),
            "conflict_rank": conflict_rank.detach(),
            "conflict_sparse": conflict_sparse.detach(),
            "risk_gap": output["risk_gap"].detach().mean(),
            "conflict_score": output["conflict_score"].detach().mean(),
        }
