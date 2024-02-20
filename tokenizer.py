import spacy
import torch

nlp = spacy.load("en_core_web_sm")


def tokenize(text: str) -> list[str]:
    return [token.text.lower() for token in nlp.tokenizer(text)]


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {}
    reverse_vocab = {}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                index = len(vocab)
                vocab[token] = index
                reverse_vocab[index] = token
    return vocab, reverse_vocab


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab[token] for token in tokenize(text)]


def decode(indices: torch.Tensor, reverse_vocab: dict[int, str]) -> str:
    return " ".join(reverse_vocab[i.item()] for i in indices)
