import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from attention import MultiHeadAttention
from tokenizer import build_vocab, encode, decode

torch.manual_seed(42)

EMBED_DIM = 128

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_txts(path: str) -> list[str]:
    """
    Read all .txt files in a directory and return a list of their contents.
    """
    files = os.listdir(path)
    assert (
        len(files) > 0
    ), "Please add training data. Move one or more .txt files inside the `data` directory."

    txt_contents = []
    for file_name in files:
        with open(os.path.join(path, file_name), "r") as file:
            txt_contents.append(file.read())
    return txt_contents


class EncodedDataset(Dataset):
    def __init__(self, encoded_txts: torch.Tensor, block_size: int = 16):
        self.encoded_txts = encoded_txts
        self.block_size = block_size

    def __len__(self):
        return self.encoded_txts.shape[0] - self.block_size

    def __getitem__(self, idx: int):
        return (
            self.encoded_txts[idx : idx + self.block_size],
            self.encoded_txts[idx + 1 : idx + self.block_size + 1],
        )


class Model(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, num_heads: int = 1, block_size: int = 16
    ):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pe_embedding = nn.Embedding(block_size, embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x) + self.pe_embedding(
            torch.arange(x.shape[1], device=device)
        )
        x = self.attention(x)
        x = self.linear(x)
        return x


def train(model: Model, dataset: Dataset, epochs: int = 10, batch_size: int = 32):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0

        for X, y in tqdm(dataloader):
            optimizer.zero_grad()
            input = X.to(device)
            y_pred = model(input)
            loss = loss_fn(y_pred.permute(0, 2, 1), y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")


def save_embeddings(model: Model, vocab: dict[str, int], path: str):
    embeddings = model.embedding.weight.detach().cpu().numpy()
    with open(path, "w") as file:
        for word, idx in vocab.items():
            file.write(f"{word}\t[{','.join([str(el) for el in embeddings[idx]])}]\n")


def generate(
    model: Model,
    vocab: dict[str, int],
    reverse_vocab: dict[int, str],
    text: str,
    max_len: int = 16,
):
    encoded_text = encode(text, vocab)
    for _ in range(max_len - len(encoded_text)):
        next_token = (
            model(
                torch.Tensor(encoded_text)
                .unsqueeze(0)
                .to(dtype=torch.long, device=device)
            )[0]
            .argmax(1)[-1]
            .item()
        )
        encoded_text.append(next_token)

    out = torch.Tensor(encoded_text).to(dtype=torch.long)
    return decode(out, reverse_vocab)


if __name__ == "__main__":
    save_path = "models/model.pt"

    txt_contents = read_txts("data/")
    vocab, reverse_vocab = build_vocab(txt_contents)

    if os.path.exists(save_path):
        model = torch.load(save_path)
    else:
        encoded_txts = (
            torch.concatenate(
                [torch.Tensor(encode(txt, vocab)) for txt in txt_contents]
            )
            .to(dtype=torch.long)
            .flatten()
        )
        dataset = EncodedDataset(encoded_txts)
        model = Model(len(vocab), EMBED_DIM).to(device)
        train(model, dataset, 10)
        torch.save(model, save_path)

    text = "The quick brown fox jumps over the lazy dog"
    print(
        generate(
            model,
            vocab,
            reverse_vocab,
            text,
        )
    )

    # save_embeddings(model, vocab, "models/embeddings.txt")
