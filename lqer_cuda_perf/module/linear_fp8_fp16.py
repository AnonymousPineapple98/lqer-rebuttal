import torch
import torch.nn as nn
from transformer_engine.pytorch.float8_tensor import Float8Tensor


class LinearFp8Fp16(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device="cuda",
        dtype=torch.float16,
        rank: int = 4,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight = nn.Parameter(
            Float8Tensor.to_float8(
                torch.zeros(
                    (out_features, in_features), dtype=torch.float16, device=device
                ),
            )
        )
        self.A = nn.Parameter(
            torch.zeros((in_features, rank), dtype=torch.float16, device=device)
        )
        self.B = nn.Parameter(
            torch.zeros((rank, out_features), dtype=torch.float16, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xw = nn.functional.linear(x, self.weight, self.bias)
        xa = torch.matmul(x, self.A)
        xab = torch.matmul(xa, self.B)
        return xw + xab
