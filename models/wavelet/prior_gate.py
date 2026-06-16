"""Attention gate for wavelet prior injection."""
import torch
import torch.nn.functional as F
from torch import nn


class PriorGate(nn.Module):
    """Inject wavelet prior into Mamba features via attention gate.

    F' = F + sigmoid(Conv(Prior)) * F
    """
    def __init__(self, feature_dim: int, prior_dim: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(prior_dim, feature_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        if prior.shape[-2:] != feature.shape[-2:]:
            prior = F.interpolate(prior, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(prior)
        return feature + gate * feature
