import torch
from torch import nn

# https://github.com/casper-hansen/AutoAWQ_kernels
import awq_ext  # with CUDA kernels


class FastRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float16, device="cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x):
        output = torch.empty_like(x)
        awq_ext.layernorm_forward_cuda(x, self.weight, output, self.variance_epsilon)

        return output
