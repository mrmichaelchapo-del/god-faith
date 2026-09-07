import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_heads,
        hidden_size,
        dropout=0.0
    ):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            embedding_size
        )

        self.attention = MultiHeadAttention(
            embedding_size=embedding_size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward_norm = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = FeedForward(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

    def forward(self, x):
        # Attention + residual connection
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Feed-forward + residual connection
        x = x + self.feed_forward(
            self.feed_forward_norm(x)
        )

        return x