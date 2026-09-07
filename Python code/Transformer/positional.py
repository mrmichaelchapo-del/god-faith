import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, max_sequence_length=4096, base=10000):
        super().__init__()

        if head_size % 2 != 0:
            raise ValueError(
                "head_size must be even for RoPE"
            )

        self.head_size = head_size
        self.max_sequence_length = max_sequence_length
        self.base = base

        # Frequencies for each pair of dimensions
        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    head_size,
                    2,
                    dtype=torch.float32
                ) / head_size
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

    def forward(self, x):
        """
        x shape:
            [batch, heads, sequence_length, head_size]
        """

        sequence_length = x.shape[-2]

        positions = torch.arange(
            sequence_length,
            device=x.device,
            dtype=self.inv_freq.dtype
        )

        frequencies = torch.outer(
            positions,
            self.inv_freq
        )

        cos = torch.cos(frequencies)
        sin = torch.sin(frequencies)

        # [sequence, head_size/2]
        return cos, sin


def rotate_half(x):
    """
    Rotates pairs of dimensions:
    [x1, x2] -> [-x2, x1]
    """

    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat