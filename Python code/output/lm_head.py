import torch
import torch.nn as nn


class LMHead(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super().__init__()

        self.projection = nn.Linear(
            embedding_size,
            vocab_size,
            bias=False
        )

    def forward(self, x):
        """
        x:
            [batch_size, sequence_length, embedding_size]

        returns:
            [batch_size, sequence_length, vocab_size]
        """

        return self.projection(x)