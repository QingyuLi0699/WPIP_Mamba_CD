"""Residual Mamba refinement for uncertain regions."""
import torch
from torch import nn
from models.backbone.mambahsi_blocks import BothMamba


class RefinementMamba(nn.Module):
    def __init__(self, embed_dim: int = 128, token_num: int = 4, group_num: int = 4):
        super().__init__()
        self.block = BothMamba(
            channels=embed_dim,
            token_num=token_num,
            use_residual=True,
            group_num=group_num,
            use_att=True,
        )
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(group_num, embed_dim), embed_dim),
            nn.SiLU(),
        )

    def forward(self, feature: torch.Tensor, uncertain_mask: torch.Tensor) -> torch.Tensor:
        refined = self.proj(self.block(feature))
        mask = uncertain_mask.to(dtype=feature.dtype)
        if mask.shape[-2:] != feature.shape[-2:]:
            raise ValueError("uncertain_mask must match feature spatial size.")
        return feature * (1.0 - mask) + refined * mask
