"""
MambaHSI backbone blocks adapted from the uploaded official MambaHSI implementation.

Original uploaded file defines SpeMamba, SpaMamba, BothMamba and MambaHSI.
This file keeps the same core logic, while exposing a feature encoder for change detection.
"""
import math
import torch
from torch import nn

try:
    from mamba_ssm import Mamba as _MambaSSM
except Exception:  # Allows import/debug on machines without mamba_ssm installed.
    _MambaSSM = None


class _FallbackMamba(nn.Module):
    """Small MLP fallback used for import/debug and CPU-only smoke tests.

    Real experiments should install and run mamba-ssm on CUDA. Some mamba-ssm
    builds import successfully but expose CUDA-only kernels, so CPU tensors are
    routed here even when the package is present.
    """
    def __init__(self, d_model, expand=2, **_):
        super().__init__()
        hidden = int(d_model * expand)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


class Mamba(nn.Module):
    """mamba-ssm wrapper with CPU-safe fallback."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.real = None
        if _MambaSSM is not None:
            self.real = _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fallback = _FallbackMamba(d_model=d_model, expand=expand)

    def forward(self, x):
        if self.real is not None and x.is_cuda:
            return self.real(x)
        return self.fallback(x)


class SpeMamba(nn.Module):
    """Spectral Mamba block.

    It groups channels into spectral tokens and applies Mamba over the token dimension.
    Input/output shape: [B, C, H, W].
    """
    def __init__(self, channels: int, token_num: int = 8, use_residual: bool = True, group_num: int = 4):
        super().__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU(),
        )

    def padding_feature(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c < self.channel_num:
            pad_c = self.channel_num - c
            pad_features = torch.zeros((b, pad_c, h, w), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad_features], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        b, h, w, c_pad = x_pad.shape
        x_flat = x_pad.view(b * h * w, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(b, h, w, c_pad).permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        x_proj = x_proj[:, :x.shape[1], :, :]
        return x + x_proj if self.use_residual else x_proj


class SpaMamba(nn.Module):
    """Spatial Mamba block.

    It flattens the spatial image sequence and applies Mamba over spatial positions.
    Input/output shape: [B, C, H, W].
    """
    def __init__(self, channels: int, use_residual: bool = True, group_num: int = 4, use_proj: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_re = x.permute(0, 2, 3, 1).contiguous()
        b, h, w, c = x_re.shape
        # Same global spatial scan behavior as the uploaded MambaHSI implementation.
        x_flat = x_re.view(1, -1, c)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        return x + x_recon if self.use_residual else x_recon


class BothMamba(nn.Module):
    """Spatial-spectral Mamba block with optional learnable fusion weights."""
    def __init__(self, channels: int, token_num: int = 4, use_residual: bool = True,
                 group_num: int = 4, use_att: bool = True):
        super().__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels, use_residual=use_residual, group_num=group_num)
        self.spe_mamba = SpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        return fusion_x + x if self.use_residual else fusion_x


class DualBranchMambaEncoder(nn.Module):
    """MambaHSI feature encoder for semantic change detection.

    Unlike the original MambaHSI classifier, this module returns deep feature maps
    instead of class logits.
    """
    def __init__(self, in_channels: int, embed_dim: int = 128, token_num: int = 4,
                 group_num: int = 4, use_residual: bool = True, use_att: bool = True,
                 depth: int = 3, downsample: bool = True, use_patch_embedding: bool = True):
        super().__init__()
        self.use_patch_embedding = use_patch_embedding
        self.downsample = downsample

        if use_patch_embedding:
            self.patch_embedding = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(group_num, embed_dim),
                nn.SiLU(),
            )
        else:
            assert in_channels == embed_dim, "If use_patch_embedding=False, in_channels must equal embed_dim."
            self.patch_embedding = nn.Identity()

        blocks = []
        for i in range(depth):
            blocks.append(BothMamba(
                channels=embed_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
            ))
            if downsample and i < depth - 1:
                blocks.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        self.mamba = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.mamba(x)
        return x
