import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import TransformerEncoderLayer
from torch import Tensor
from torch.nn import TransformerEncoder
import math

class Embedding(nn.Module):
    def __init__(self, tokens, dims, dtype=None):
        super().__init__()
        self.linear = nn.Linear(tokens, dims, dtype=dtype, bias=False)
        self.tokens = tokens
        self.dtype = dtype

    def forward(self, x):
        return self.linear(F.one_hot(x, self.tokens).to(self.dtype))

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, dtype=None, device=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, dtype=dtype)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = Embedding(ntoken, d_model, dtype=dtype)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken, **factory_kwargs)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.linear.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = src.type(torch.cfloat)

        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
