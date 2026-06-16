from torch import nn


class SemanticEmbeddingHead(nn.Module):
    """Project shared change feature into semantic embedding space for prototypes."""
    def __init__(self, in_channels: int, embed_dim: int = 128, group_num: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(group_num, embed_dim), embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(group_num, embed_dim), embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.proj(x)
