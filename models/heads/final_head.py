from torch import nn


class FinalSemanticHead(nn.Module):
    """Final K+1 semantic change classification head.

    Class 0 is no-change. Classes 1..K are semantic change classes.
    """
    def __init__(self, embed_dim: int, num_total_classes: int, group_num: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(group_num, embed_dim), embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, num_total_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)
