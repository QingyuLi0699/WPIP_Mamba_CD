from torch import nn


class BinaryHead(nn.Module):
    """Coarse binary change detection head: no-change vs change."""
    def __init__(self, in_channels: int, hidden_dim: int = 128, group_num: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(group_num, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)
