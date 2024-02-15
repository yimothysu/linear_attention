import spacy

nlp = spacy.load("en_core_web_sm")

def tokenize(text: str) -> list[str]:
    return [token.text for token in nlp.tokenizer(text)]

def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def encode(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab[token] for token in tokenize(text)]

def decode(indices: list[int], vocab: dict[str, int]) -> str:
    return " ".join(vocab[i] for i in indices)
