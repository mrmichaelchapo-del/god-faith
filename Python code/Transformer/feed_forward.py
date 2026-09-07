import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        dropout=0.0
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_size,
                hidden_size
            ),

            nn.GELU(),

            nn.Linear(
                hidden_size,
                embedding_size
            ),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)