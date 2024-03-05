import os

from model import Model


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


def save_embeddings(model: Model, vocab: dict[str, int], path: str):
    embeddings = model.embedding.weight.detach().cpu().numpy()
    with open(path, "w") as file:
        for word, idx in vocab.items():
            file.write(f"{word}\t[{','.join([str(el) for el in embeddings[idx]])}]\n")
