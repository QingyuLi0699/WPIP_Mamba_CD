"""Difference fusion for dual-temporal features."""
import torch
from torch import nn


class DifferenceFusion(nn.Module):
    """Fuse absolute difference and correlation features.

    F_cd = Conv([|F1-F2|, F1*F2])
    """
    def __init__(self, in_dim: int, out_dim: int = None, group_num: int = 4):
        super().__init__()
        out_dim = out_dim or in_dim * 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(group_num, out_dim), out_dim),
            nn.SiLU(),
        )

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(feat_t1 - feat_t2)
        corr = feat_t1 * feat_t2
        return self.proj(torch.cat([diff, corr], dim=1))
