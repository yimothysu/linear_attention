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
    txt_contents = []
    for file_name in os.listdir(path):
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
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int = 1):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.attention(x, x, x)
        x = self.linear(x)
        return x


def train(model: Model, dataset: Dataset, epochs: int = 10, batch_size: int = 32):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()
            input = batch[0].to(device)
            target = input.clone().detach()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}")


if __name__ == "__main__":
    txt_contents = read_txts("data/")
    vocab = build_vocab(txt_contents)
    encoded_txts = torch.Tensor([encode(txt, vocab) for txt in txt_contents]).flatten()
    dataset = EncodedDataset(encoded_txts)
    model = Model(len(vocab), EMBED_DIM).to(device)
    train(model, dataset)
    text = "The quick brown fox jumps over the lazy dog"
    encoded_text = encode(text, vocab)
    print(decode(model(torch.Tensor(encoded_text).to(device)).argmax(1), vocab))
