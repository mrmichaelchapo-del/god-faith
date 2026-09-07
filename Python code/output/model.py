import math
import torch
import torch.nn as nn

from .positional import RotaryEmbedding, apply_rotary_embedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_heads,
        max_sequence_length=4096,
        dropout=0.0
    ):
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

        self.rope = RotaryEmbedding(
            head_size=self.head_size,
            max_sequence_length=max_sequence_length
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

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

        cos, sin = self.rope(q)

        q = apply_rotary_embedding(
            q,
            cos,
            sin
        )

        k = apply_rotary_embedding(
            k,
            cos,
            sin
        )

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_size)

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

        attention = torch.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(attention)

        output = torch.matmul(
            attention,
            v
        )

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            self.embedding_size
        )

        return self.out_proj(output)