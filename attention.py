import plumax

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        softmax_fn: nn.Module = nn.Softmax(dim=-1),
        is_masked: bool = False,
    ):
        """
        d_k and d_v are per-head dimensions. If not provided, they default to embed_dim // num_heads.
        """

        super(MultiHeadAttention, self).__init__()
        self.softmax_fn = softmax_fn
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.is_masked = is_masked
        if is_masked:
            self.register_buffer(
                "mask",
                torch.tril(
                    torch.ones((self.head_dim, self.head_dim), dtype=torch.bool)
                ),
            )

        self.d_k = d_k if d_k is not None else self.head_dim
        self.d_v = d_v if d_v is not None else self.head_dim

        self.q_proj_weight = nn.Linear(embed_dim, self.d_k * num_heads, bias=False)
        self.k_proj_weight = nn.Linear(embed_dim, self.d_k * num_heads, bias=False)
        self.v_proj_weight = nn.Linear(embed_dim, self.d_v * num_heads, bias=False)
        self.fc_out = nn.Linear(num_heads * self.d_v, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, l, _ = x.shape

        q: torch.Tensor = self.q_proj_weight(x)
        k: torch.Tensor = self.k_proj_weight(x)
        v: torch.Tensor = self.v_proj_weight(x)

        q = q.view(N, self.num_heads, l, self.d_k)
        k = k.view(N, self.num_heads, l, self.d_k)
        v = v.view(N, self.num_heads, l, self.d_v)

        attention_unnormalized: torch.Tensor = q @ k.transpose(-2, -1) / self.d_k**0.5
        if self.is_masked:
            attention_unnormalized.masked_fill_(self.mask, float("-inf"))
        attention = self.softmax_fn(attention_unnormalized)
        out = (attention @ v).view(N, l, -1)
        out = self.fc_out(out)

        return out

    def set_softmax_fn(self, softmax_fn):
        self.softmax_fn = softmax_fn


class PLUMax(nn.Module):
    def __init__(self, k=10_000):
        super(PLUMax, self).__init__()
        self.plu = plumax.exp_to_plu(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(plumax.plu_max(self.plu, x.detach().numpy())).to(
            torch.float32
        )


if __name__ == "__main__":
    att = MultiHeadAttention(16, 4)
    x = torch.rand(1, 5, 16)
    out_softmax = att(x)
    att.set_softmax_fn(PLUMax())
    out_plumax = att(x)

    print("Output of Attention, normalizing via softmax")
    print(out_softmax)
    print("-" * 20)
    print("Output of Attention, normalizing via plumax")
    print(out_plumax)
    print("-" * 20)
    print(
        "Frobenius norm of difference between softmax and plumax outputs:",
        torch.norm(out_softmax - out_plumax).item(),
    )
