import torch

from model import Model
from tokenizer import decode, encode


def generate(
    model: Model,
    vocab: dict[str, int],
    reverse_vocab: dict[int, str],
    text: str,
    device: str,
    max_len: int = 16,
):
    encoded_text = encode(text, vocab)
    for _ in range(max_len - len(encoded_text)):
        token_probs: torch.Tensor = model(
            torch.Tensor(encoded_text).unsqueeze(0).to(dtype=torch.long, device=device)
        )[0]
        next_token = token_probs.argmax(dim=1)[-1].item()
        encoded_text.append(next_token)

    out = torch.Tensor(encoded_text).to(dtype=torch.long)
    return decode(out, reverse_vocab)
