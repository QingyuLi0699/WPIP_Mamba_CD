"""Prototype assignment and pseudo-label generation."""
import torch
import torch.nn.functional as F
from torch import nn


class PrototypeAssignment(nn.Module):
    def __init__(self, temperature: float = 0.1, pseudo_threshold: float = 0.9):
        super().__init__()
        self.temperature = temperature
        self.pseudo_threshold = pseudo_threshold

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor):
        """Compute prototype logits for each pixel.

        Args:
            features:   [B, D, H, W]
            prototypes: [K, D]
        Returns:
            proto_logits: [B, K, H, W]
            proto_conf:   [B, 1, H, W]
            pseudo_label: [B, H, W] with values 1..K
            pseudo_mask:  [B, H, W] bool
        """
        b, d, h, w = features.shape
        feat = F.normalize(features, dim=1)
        proto = F.normalize(prototypes, dim=1)
        logits = torch.einsum("bdhw,kd->bkhw", feat, proto) / self.temperature
        prob = torch.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1, keepdim=True)
        pseudo_label = pred.squeeze(1) + 1  # map prototype index 0..K-1 to semantic labels 1..K
        pseudo_mask = conf.squeeze(1) > self.pseudo_threshold
        return logits, conf, pseudo_label, pseudo_mask
