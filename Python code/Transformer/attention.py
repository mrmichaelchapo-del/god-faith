import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout=0.0):
        super().__init__()

        if embedding_size % num_heads != 0:
            raise ValueError(
                "embedding_size must be divisible by num_heads"
            )

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        self.q_proj = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.k_proj = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.v_proj = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.out_proj = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x:
            [batch_size, sequence_length, embedding_size]
        """

        batch_size, sequence_length, _ = x.shape

        # Create queries, keys and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into attention heads
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_size)

        # Causal mask
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        # Convert scores to probabilities
        attention = torch.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(attention)

        # Apply attention to values
        output = torch.matmul(
            attention,
            v
        )

        # Combine heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            self.embedding_size
        )

        # Final projection
        return self.out_proj(output)