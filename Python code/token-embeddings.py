import numpy as np


class TokenEmbedding:
    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # One vector for every token
        self.weights = np.random.randn(
            vocab_size,
            embedding_size
        ) * 0.02

    def forward(self, token_ids):
        return self.weights[token_ids]


# Example
vocab_size = 10000
embedding_size = 256

embedding = TokenEmbedding(
    vocab_size,
    embedding_size
)

# Token IDs from the tokenizer
tokens = np.array([4, 15, 92, 301])

vectors = embedding.forward(tokens)

print("Token IDs:")
print(tokens)

print("\nEmbedding shape:")
print(vectors.shape)

print("\nEmbeddings:")
print(vectors)