import torch


def gen_dft_matrix(mask: torch.Tensor) -> torch.Tensor:
    N = mask.shape[0]
    factor = torch.exp(torch.tensor(-1j * 2 * torch.pi / N))
    u = torch.arange(0., N)
    powers = torch.outer(u, u)
    mat = torch.pow(factor, powers)
    return mat * mask


def masked_fft(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    dft_matrix = gen_dft_matrix(mask)
    return torch.einsum('sd,sbf->dbf', dft_matrix, x)
