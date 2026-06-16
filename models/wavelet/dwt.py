"""Differentiable 2D Haar DWT for feature maps."""
import torch
import torch.nn.functional as F
from torch import nn


class HaarDWT2D(nn.Module):
    """Single-level Haar DWT.

    Input:  [B, C, H, W]
    Output: LL, LH, HL, HH with shape [B, C, ceil(H/2), ceil(W/2)].
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh
