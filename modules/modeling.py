# Complex Neural Module implementation
# coding = utf-8
# Copyright 2025 Rikka Botan. All rights reserved
# Licensed under "MIT License"

import torch
from torch import nn
from typing import Any
from einops import rearrange
import torch.nn.functional as F

def BotanCC(
    ReIm: torch.Tensor
) -> torch.Tensor:
    """
    ## Complex Conjugate function

    x + yi => x - yi

    ReIm[bsz, seql, 2, embs] 

    ReIm[..., 0, :]: Real part,
    ReIm[..., 1, :]: Imaginary part
    """
    ReIm_clone = ReIm.clone()
    ReIm_clone[..., 1, :] = -ReIm_clone[..., 1, :]
    return ReIm_clone

def BotanCM(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    ## Complex Multiplication function

    (a + bi)(c + di) = (ac - bd) + (ad + bc)i

    Any[..., 0, :]: Real part, 
    Any[..., 1, :]: Imaginary part
    """
    ac_bd = x * BotanCC(y)
    adbc = x * y[..., ::1, :]
    return torch.stack([ac_bd.sum(-2), adbc.sum(-2)], dim=-2)


class ComplexLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Any = None,
        dtype: Any = None
    ):
        """
        ## Linear class on Complex Plane
        """
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError("out_features should be a multiple of 2.")
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features*2,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.linear(hidden_states)
        hidden_states = rearrange(hidden_states, "... (c e) -> ... c e", c=4)
        c1, c2 = torch.chunk(hidden_states, chunks=2, dim=-2)
        hidden_states = BotanCM(c1, c2)
        hidden_states = rearrange(hidden_states, "... c e -> ... (c e)", c=2)
        return hidden_states


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        ## Convolution 2D class on Complex Plane
        """
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError("out_channels should be a multiple of 2.")
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels*2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = rearrange(hidden_states, "n (c e) h w -> n h w c e", c=4)
        c1, c2 = torch.chunk(hidden_states, chunks=2, dim=-2)
        hidden_states = BotanCM(c1, c2)
        hidden_states = rearrange(hidden_states, "n h w c e -> n (c e) h w", c=2)
        return hidden_states


class ComplexAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        bias: bool = False,
        device: str = None,
        dtype: str = None
    ):
        """
        ## Complex Attention implementation
        """
        super().__init__()
        self.qkv_linear = ComplexLinear(
            in_features=hidden_size,
            out_features=hidden_size*3,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.out_linear = ComplexLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.heads = heads
        self.scale = hidden_size**0.5

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        bsz, seql, _ = hidden_states.size()
        QKV = self.qkv_linear(hidden_states)
        QKV = QKV.reshape(bsz, seql, self.heads, -1).transpose(1, 2)
        Q, K, V = torch.chunk(QKV, chunks=3, dim=-1)
        matrix = torch.matmul(Q, K.transpose(2, 3)) / self.scale
        matrix = F.softmax(matrix, dim=-1)
        outputs = torch.matmul(matrix, V)
        outputs = self.out_linear(outputs.transpose(1, 2).reshape(bsz, seql, -1))
        return outputs

