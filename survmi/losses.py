from typing import Optional

import torch
import torch.nn.functional as F


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log likelihood.

    Larger risk means shorter survival. `event` should be 1 for observed events
    and 0 for censored samples.
    """

    risk = risk.reshape(-1)
    time = time.reshape(-1)
    event = event.reshape(-1).float()

    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_risk_set = torch.logcumsumexp(risk, dim=0)
    per_sample = risk - log_risk_set
    observed = event.sum().clamp_min(1.0)
    return -(per_sample * event).sum() / observed


def survival_risk_group(
    time: torch.Tensor,
    event: torch.Tensor,
    cutoff: Optional[float | torch.Tensor] = None,
) -> torch.Tensor:
    """Build coarse risk groups for stable survival-aware contrastive learning.

    This is not an exact conditional-MI estimator. It provides a practical
    risk-aware negative sampler: observed short-survival patients are high risk,
    observed long-survival patients are lower risk, and censored patients are
    treated conservatively according to their observed time.
    """

    time = time.float().reshape(-1)
    event = event.float().reshape(-1)
    if cutoff is None:
        observed = time[event > 0]
        if observed.numel() == 0:
            cutoff_t = time.median()
        else:
            cutoff_t = observed.median()
    else:
        cutoff_t = torch.as_tensor(cutoff, dtype=time.dtype, device=time.device)
    high = (time <= cutoff_t) & (event > 0)
    low = time > cutoff_t
    group = torch.ones_like(time, dtype=torch.long)
    group[high] = 2
    group[low] = 0
    return group


def risk_aware_info_nce_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    cutoff: Optional[float | torch.Tensor] = None,
    temperature: float = 0.2,
    hard_negative_weight: float = 2.0,
    soft_negative_weight: float = 0.5,
) -> torch.Tensor:
    """Symmetric InfoNCE with survival-risk-aware negative weighting.

    Negatives from different coarse risk groups are emphasized. Negatives from
    the same risk group are down-weighted, which is a stable approximation of
    survival-conditioned contrast without directly estimating I(.;.|Y).
    """

    if query.shape != key.shape:
        raise ValueError(f"query and key must have same shape, got {query.shape} and {key.shape}")
    batch = query.size(0)
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = query @ key.t() / temperature
    labels = torch.arange(batch, device=query.device)

    group = survival_risk_group(time, event, cutoff=cutoff).to(query.device)
    different_group = group[:, None] != group[None, :]
    weights = torch.where(
        different_group,
        torch.full_like(logits, hard_negative_weight),
        torch.full_like(logits, soft_negative_weight),
    )
    weights.fill_diagonal_(1.0)
    weighted_logits = logits + weights.clamp_min(1e-6).log()

    return 0.5 * (
        F.cross_entropy(weighted_logits, labels)
        + F.cross_entropy(weighted_logits.t(), labels)
    )


def pair_level_patch_pathway_nce_loss(
    proto_tokens: torch.Tensor,
    pathway_tokens: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    cutoff: Optional[float | torch.Tensor] = None,
    temperature: float = 0.2,
    topk: int = 4,
    max_negatives: int = 16,
    hard_negative_weight: float = 2.0,
    soft_negative_weight: float = 0.5,
) -> torch.Tensor:
    """Risk-aware contrast directly over prototype x pathway token pairs.

    For each query patient, the positive logit uses all within-patient
    prototype-pathway pairs. Negatives are sampled cross-patient pairs and
    weighted by coarse survival-risk groups. This keeps the signal pair-level
    while avoiding the O(B^2*K*M) memory cost of exhaustive cross-patient pairs.
    """

    if proto_tokens.dim() != 3 or pathway_tokens.dim() != 3:
        raise ValueError("proto_tokens and pathway_tokens must be [batch, tokens, dim]")
    if proto_tokens.size(0) != pathway_tokens.size(0):
        raise ValueError("proto_tokens and pathway_tokens must share batch size")
    if proto_tokens.size(-1) != pathway_tokens.size(-1):
        raise ValueError("proto_tokens and pathway_tokens must share hidden dim")

    batch = proto_tokens.size(0)
    proto = F.normalize(proto_tokens, dim=-1)
    pathway = F.normalize(pathway_tokens, dim=-1)

    pos_scores = torch.einsum("bkd,bmd->bkm", proto, pathway).flatten(start_dim=1)
    k = min(topk, pos_scores.size(-1))
    pos_logits = pos_scores.topk(k=k, dim=-1).values.mean(dim=-1, keepdim=True) / temperature

    neg_count = min(max_negatives, batch - 1)
    if neg_count <= 0:
        return (proto.sum() + pathway.sum()) * 0.0

    neg_indices = []
    all_indices = torch.arange(batch, device=proto.device)
    for anchor in range(batch):
        candidates = all_indices[all_indices != anchor]
        if candidates.numel() > neg_count:
            candidates = candidates[torch.randperm(candidates.numel(), device=proto.device)[:neg_count]]
        neg_indices.append(candidates)
    neg_indices = torch.stack(neg_indices, dim=0)

    neg_pathway = pathway[neg_indices]
    neg_scores = torch.einsum("bkd,bnmd->bnkm", proto, neg_pathway).flatten(start_dim=2)
    neg_logits = neg_scores.topk(k=k, dim=-1).values.mean(dim=-1) / temperature

    group = survival_risk_group(time, event, cutoff=cutoff).to(proto.device)
    neg_groups = group[neg_indices]
    different_group = group[:, None] != neg_groups
    weights = torch.where(
        different_group,
        torch.full_like(neg_logits, hard_negative_weight),
        torch.full_like(neg_logits, soft_negative_weight),
    )
    logits = torch.cat([pos_logits, neg_logits + weights.clamp_min(1e-6).log()], dim=1)

    neg_proto = proto[neg_indices]
    rev_scores = torch.einsum("bmd,bnkd->bnmk", pathway, neg_proto).flatten(start_dim=2)
    rev_neg_logits = rev_scores.topk(k=k, dim=-1).values.mean(dim=-1) / temperature
    rev_logits = torch.cat([pos_logits, rev_neg_logits + weights.clamp_min(1e-6).log()], dim=1)

    labels = torch.zeros(batch, dtype=torch.long, device=proto.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(rev_logits, labels))


def orthogonality_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """惩罚两组 token 嵌入之间的对齐度（余弦平方），使 b 子空间与 a 正交。

    用于把 conflict 通道约束为 agreement 通道的*残差*：agreement 解释掉的一致信息，
    conflict 不再重复捕获，从而 conflict 只保留 agreement 无法解释的跨模态分歧。
    a, b: [..., dim]，在最后一维上比较方向。
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    cos = (a * b).sum(dim=-1)
    return cos.pow(2).mean()


def pairwise_rank_loss(score: torch.Tensor, target: torch.Tensor, margin: float = 1e-4) -> torch.Tensor:
    """成对排序损失：让 score 的大小顺序与 target 一致（target 越大 score 越大）。

    这里用于把 conflict 分数对齐到两个模态风险头的分歧 |wsi_risk - omics_risk|，
    使 conflict 分数成为"跨模态预后分歧强度"的可解释度量。
    """
    score = score.reshape(-1)
    target = target.reshape(-1)
    target_diff = target[:, None] - target[None, :]
    valid = target_diff > margin
    if not valid.any():
        return score.sum() * 0.0
    score_diff = score[:, None] - score[None, :]
    return F.softplus(-score_diff[valid]).mean()
