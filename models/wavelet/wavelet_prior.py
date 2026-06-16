"""Wavelet prior generator for dual-temporal HSI features."""
import torch
from torch import nn
from .dwt import HaarDWT2D


class WaveletPriorGenerator(nn.Module):
    """Generate a learnable wavelet change prior from shallow feature maps.

    P_sem = |LL1 - LL2|
    P_str = concat(|LH1-LH2|, |HL1-HL2|, |HH1-HH2|)
    prior = Conv([P_sem, P_str])
    """
    def __init__(self, in_channels: int, prior_dim: int = 32, group_num: int = 4):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.prior_net = nn.Sequential(
            nn.Conv2d(in_channels * 4, prior_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(group_num, prior_dim), prior_dim),
            nn.SiLU(),
            nn.Conv2d(prior_dim, prior_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(group_num, prior_dim), prior_dim),
            nn.SiLU(),
        )

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor) -> torch.Tensor:
        ll1, lh1, hl1, hh1 = self.dwt(feat_t1)
        ll2, lh2, hl2, hh2 = self.dwt(feat_t2)

        p_sem = torch.abs(ll1 - ll2)
        p_lh = torch.abs(lh1 - lh2)
        p_hl = torch.abs(hl1 - hl2)
        p_hh = torch.abs(hh1 - hh2)
        prior = torch.cat([p_sem, p_lh, p_hl, p_hh], dim=1)
        return self.prior_net(prior)
