import torch
from torch import nn
from .linear_fp8_fp16 import LinearFp8Fp16
from .norm import FastRMSNorm


def get_linear(
    name,
    in_features,
    out_features,
    bias: bool = False,
    device="cuda",
    dtype=torch.float16,
    rank: int = 4,
):
    match name:
        case "fp8fp16":
            return LinearFp8Fp16(in_features, out_features, bias, device, dtype, rank)
        case "fp16":
            return nn.Linear(in_features, out_features, bias, device, dtype)
        case _:
            raise RuntimeError(f"unknown linear type {name}")
