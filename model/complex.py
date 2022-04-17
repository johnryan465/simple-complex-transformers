from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fft import masked_fft


class FourierAttention(nn.Module):
    def __init__(self, batch_first: bool, n_heads: int, dtype, *args, **kwargs):
        super().__init__()
        self.heads = n_heads
        self.dtype = dtype
        self.feature_dim = -1
        self.sequence_dim = -3
        if batch_first:
            self.sequence_dim = -2

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        s = x.shape
        # x = x.view(*s[:-1], -1, self.heads)
        x = torch.fft.fft(masked_fft(x, mask, dim=self.sequence_dim), dim=self.feature_dim, norm="ortho")
        # x = torch.flatten(x, start_dim=-2)
        if self.dtype == torch.cfloat:
            return x
        else:
            return x.real


class ComplexDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if x.is_complex():
            mask = F.dropout(torch.ones_like(x.real), self.p)
            return x * mask
        else:
            return F.dropout(x, self.p)


def custom_layer_norm(
    x: torch.Tensor, dim: Tuple[int] = (-1,), eps: float = 0.00001
) -> torch.Tensor:
    mean = torch.mean(x, dim=dim, keepdim=True)
    # print(mean.shape)
    s = x - mean
    # print(s.shape)
    var = (s * torch.conj(s)).mean(dim=(-1), keepdim=True)
    # print(var.shape)
    return (x - mean) / torch.sqrt(var + eps)


class ComplexLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, *args, **kwargs):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return custom_layer_norm(x, eps=self.eps)
