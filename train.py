import argparse
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data import read_txts

from model import Model
from tokenizer import build_vocab, encode

torch.manual_seed(42)

EMBED_DIM = 128

device = "cuda" if torch.cuda.is_available() else "cpu"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        action="store_true",
        default=False,
        help="Load model from checkpoint",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default="models/model.pt",
        help="Path to load model from",
    )
    parser.add_argument(
        "--save_path", type=str, default="models/model.pt", help="Path to save model to"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    args = parser.parse_args()
    load = args.load
    load_path = args.load_path
    save_path = args.save_path
    num_epochs = args.epochs

    txt_contents = read_txts("data/")
    vocab, reverse_vocab = build_vocab(txt_contents)

    model = Model(len(vocab), EMBED_DIM).to(device)
    if load:
        print(f"Loading model from {load_path}.")
        model.load_state_dict(torch.load(load_path))

    encoded_txts = (
        torch.concatenate([torch.Tensor(encode(txt, vocab)) for txt in txt_contents])
        .to(dtype=torch.long)
        .flatten()
    )
    dataset = EncodedDataset(encoded_txts)
    print("Training model...")
    train(model, dataset, num_epochs)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.")

    # save_embeddings(model, vocab, "models/embeddings.txt")
