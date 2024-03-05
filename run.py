print("Loading...")

import argparse

import torch

from data import read_txts, save_embeddings
from generate import generate
from model import Model
from tokenizer import build_vocab
from train import EMBED_DIM

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_path",
    type=str,
    default="models/model.pt",
    help="Path to load model from",
)
parser.add_argument(
    "--embeddings",
    action="store_true",
    default=False,
    help="Save word embeddings to file",
)
args = parser.parse_args()
load_path = args.load_path

txt_contents = read_txts("data/")
vocab, reverse_vocab = build_vocab(txt_contents)
model = Model(len(vocab), EMBED_DIM).to(device)
model.load_state_dict(torch.load(load_path))
model.eval()
print(f"Model loaded from {load_path}.")
if args.embeddings:
    save_embeddings(model, vocab, "models/embeddings.txt")
    print("Embeddings saved to models/embeddings.txt.")

exit = False
while not exit:
    text = input("Enter a prompt: ")
    if text == "exit":
        exit = True
        continue

    completion = generate(
        model,
        vocab,
        reverse_vocab,
        text,
        device=device,
    )
    print("Completion:", completion)
