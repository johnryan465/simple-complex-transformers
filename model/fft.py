import torch
from torch.cuda import device
import math


def circulant(tensor, dim=0):
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,)).T


def gen_dft_matrix(mask: torch.Tensor) -> torch.Tensor:
    N = mask.shape[0]
    factor = torch.exp(torch.tensor(-1j * 2 * torch.pi)).to("cuda")
    u = circulant(torch.arange(0., N, device="cuda"))
    w = (torch.arange(0., N, device="cuda") / torch.arange(1., N+1, device="cuda")).unsqueeze(-1)

    powers = w @ u
    mat = torch.pow(factor, powers)
    return mat * mask


def masked_fft(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    dft_matrix = gen_dft_matrix(mask)
    N = mask.shape[0]
    return (1. / math.sqrt(N)) * torch.einsum('sd,dbf->sbf', dft_matrix, x)
