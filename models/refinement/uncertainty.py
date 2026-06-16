"""Entropy-based confidence partition for coarse binary predictions."""
import torch
import torch.nn.functional as F
from torch import nn


class ConfidencePartition(nn.Module):
    def __init__(self, entropy_threshold: float = 0.3, change_threshold: float = 0.5):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.change_threshold = change_threshold

    def forward(self, binary_logits: torch.Tensor):
        """Return reliable/uncertain masks from binary logits.

        Args:
            binary_logits: [B, 2, H, W]
        Returns:
            reliable_change:    [B, 1, H, W] bool
            reliable_nochange:  [B, 1, H, W] bool
            uncertain_mask:     [B, 1, H, W] bool
            entropy:            [B, 1, H, W]
        """
        prob = torch.softmax(binary_logits, dim=1)
        entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        change_prob = prob[:, 1:2]
        reliable = entropy < self.entropy_threshold
        reliable_change = reliable & (change_prob > self.change_threshold)
        reliable_nochange = reliable & (change_prob <= self.change_threshold)
        uncertain_mask = ~reliable
        return reliable_change, reliable_nochange, uncertain_mask, entropy
