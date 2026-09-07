import torch.nn as nn

from .block import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        embedding_size,
        num_heads,
        hidden_size,
        dropout=0.0
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
                hidden_size=hidden_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            embedding_size
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.final_norm(x)