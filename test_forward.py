"""Minimal forward test for WPIP-Mamba.
Run:
    python test_forward.py
"""
import torch
from models.wpip_mamba import WPIPMambaCD


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, c, h, w = 1, 128, 32, 32
    k = 4
    x1 = torch.randn(b, c, h, w).to(device)
    x2 = torch.randn(b, c, h, w).to(device)
    labels = torch.randint(0, k + 1, (b, h, w)).to(device)
    labels[:, :4, :4] = -1

    model = WPIPMambaCD(in_channels=c, num_change_classes=k, embed_dim=64, encoder_downsample=True).to(device)
    model.train()
    out = model(x1, x2, labels=labels, update_prototype=True)
    for key, value in out.items():
        if torch.is_tensor(value):
            print(key, tuple(value.shape), value.dtype)


if __name__ == "__main__":
    main()
