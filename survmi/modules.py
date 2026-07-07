import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphologyPrototypePooler(nn.Module):
    """Summarize many WSI patches into a fixed set of morphology prototypes."""

    def __init__(self, in_dim: int, hidden_dim: int, num_prototypes: int):
        super().__init__()
        self.patch_proj = nn.Linear(in_dim, hidden_dim)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim) / math.sqrt(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(self.patch_proj(patches))
        proto = F.normalize(self.prototypes, dim=-1)
        h_norm = F.normalize(h, dim=-1)
        logits = torch.einsum("kd,bnd->bkn", proto, h_norm) / math.sqrt(h.size(-1))
        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, :].bool(), -torch.finfo(logits.dtype).max)
        assign = torch.softmax(logits, dim=-1)
        tokens = torch.einsum("bkn,bnd->bkd", assign, h)
        return tokens, assign


class PathwayProjector(nn.Module):
    """Project gene expression vectors into biological pathway tokens."""

    def __init__(
        self,
        gene_dim: int,
        pathway_dim: int,
        hidden_dim: int,
        num_pathways: int,
        pathway_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        if pathway_mask is None:
            self.pathway_weight = nn.Parameter(torch.randn(num_pathways, gene_dim) * 0.02)
            self.register_buffer("pathway_mask", None)
        else:
            if pathway_mask.shape != (num_pathways, gene_dim):
                raise ValueError("pathway_mask must have shape [num_pathways, gene_dim]")
            self.pathway_weight = nn.Parameter(pathway_mask.float())
            self.register_buffer("pathway_mask", pathway_mask.float())
        self.token_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pathway_dim),
        )
        self.pathway_embed = nn.Parameter(torch.randn(num_pathways, pathway_dim) * 0.02)

    def forward(self, omics: torch.Tensor) -> torch.Tensor:
        weight = self.pathway_weight
        if self.pathway_mask is not None:
            weight = weight * self.pathway_mask
        denom = weight.abs().sum(dim=-1).clamp_min(1.0)
        pathway_activity = omics @ weight.t() / denom
        activity_token = self.token_mlp(pathway_activity.unsqueeze(-1))
        return activity_token + self.pathway_embed.unsqueeze(0)


class MIGatedCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mi_gate: float = 1.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mi_gate = nn.Parameter(torch.tensor(float(mi_gate)))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_tokens: torch.Tensor,
        mi_matrix: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = query_tokens.shape
        k_len = key_tokens.size(1)

        q = self.q(query_tokens).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(key_tokens).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(key_tokens).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mi_matrix is not None:
            logits = logits + F.softplus(self.mi_gate) * mi_matrix[:, None, :, :]
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, q_len, self.dim)
        return self.out(out), attn.mean(dim=1)
