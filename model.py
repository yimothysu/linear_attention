import torch
from torch import nn

from attention import MultiHeadAttention


class Model(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, num_heads: int = 1, block_size: int = 16
    ):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pe_embedding = nn.Embedding(block_size, embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, is_masked=True)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x) + self.pe_embedding(
            torch.arange(x.shape[1], device=x.device)
        )
        x = self.attention(x)
        x = self.linear(x)
        return x


# Pick patch embeddings of an image instead of word embeddings (e.g., 32x32 overlapping patches)
# Build a dicitonary out of the patches
# Many patches would be similar to each other, so they would have high cosine similarity
# Use grayscale large (e.g., 512x512 or 1024x1024) images
# Trying to discover a K-dimensional latent space
# Attention will take 32x32 patches and examine
# Decoder is reconstruction of original image
# Will yield weighted memberships of patches that minimize L2 error
# Someone has probably done this (google "single head attention autoencoder")

# 784 -> 100 -> 784
# Attention: 100 -> 50, but the 50-dim embeddings
# Decoder: 50 -> 100 using FC
# Learn a single FC layer as an autoencoder
# Single FC layer as decoder
# Then, take the 100-dim embedding and use attention to find similar images
# Attention will find memberships in the dictionary
# Ensure that two similar images have similar embeddings
# Fashion MNIST is 28x28, 60,000 instances
# 28x28 = 784 -> 100, learn an autoencoder
# Use pixel-wise L2 error, replace or add an attention head

# Self attention is a bad thing to do if there's bad embeddings for words to begin with
# But try someone else's single head attention

# Look at sequence to sequence translation
