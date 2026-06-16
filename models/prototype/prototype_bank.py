"""K-prototype memory bank for semantic change classes."""
import torch
import torch.nn.functional as F
from torch import nn


class PrototypeBank(nn.Module):
    """One prototype for each semantic change class.

    Prototypes correspond to labels 1..K in the final semantic map.
    No-change class 0 does not have a semantic prototype in this V1 design.
    """
    def __init__(self, num_change_classes: int, feat_dim: int, momentum: float = 0.99):
        super().__init__()
        self.num_change_classes = num_change_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.register_buffer("prototypes", F.normalize(torch.randn(num_change_classes, feat_dim), dim=1))
        self.register_buffer("initialized", torch.zeros(num_change_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1):
        """EMA update from labeled semantic change pixels.

        Args:
            features: [B, D, H, W]
            labels:   [B, H, W], values: 0 no-change, 1..K change classes, ignore_index ignored
        """
        if labels.shape[-2:] != features.shape[-2:]:
            raise ValueError("labels must be resized to feature spatial size before PrototypeBank.update().")

        feat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        lab = labels.reshape(-1)
        feat = F.normalize(feat, dim=1)

        for cls in range(1, self.num_change_classes + 1):
            mask = lab == cls
            if mask.any():
                mean_feat = F.normalize(feat[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                idx = cls - 1
                if not self.initialized[idx]:
                    self.prototypes[idx] = mean_feat
                    self.initialized[idx] = True
                else:
                    updated = self.momentum * self.prototypes[idx] + (1.0 - self.momentum) * mean_feat
                    self.prototypes[idx] = F.normalize(updated, dim=0)

    def get(self):
        return F.normalize(self.prototypes, dim=1)
