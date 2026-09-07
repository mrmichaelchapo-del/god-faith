import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, embedding_size, eps=1e-5):
        super().__init__()

        self.norm = nn.LayerNorm(
            embedding_size,
            eps=eps
        )

    def forward(self, x):
        return self.norm(x)