class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, text):
        # Split text into words and punctuation
        import re

        tokens = re.findall(r"\w+|[^\w\s]", text.lower())

        for token in tokens:
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.reverse_vocab[token_id] = token

    def encode(self, text):
        import re

        tokens = re.findall(r"\w+|[^\w\s]", text.lower())

        return [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokens
        ]

    def decode(self, ids):
        tokens = [
            self.reverse_vocab.get(token_id, "<UNK>")
            for token_id in ids
        ]

        text = " ".join(tokens)

        # Fix spacing around punctuation
        import re
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        return text


# Example
tokenizer = SimpleTokenizer()

tokenizer.train(
    "Hello world! This is my language model."
)

encoded = tokenizer.encode(
    "Hello world!"
)

print("Token IDs:", encoded)

decoded = tokenizer.decode(encoded)

print("Decoded:", decoded)