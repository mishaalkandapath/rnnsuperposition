from typing import Any
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

#code adapted from https://github.com/safety-research/circuit-tracer/blob/main/
def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)

class jumprelu(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        return (x * (x > torch.exp(threshold))).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


class JumpReLU(torch.nn.Module):
    def __init__(self, threshold: torch.Tensor, bandwidth: float = 2) -> None:
        super().__init__()
        self.threshold = nn.Parameter(threshold)
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return jumprelu.apply(x, self.threshold, self.bandwidth)  # type: ignore

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, bandwidth={self.bandwidth}"
    

def set_transcoder_weights(p=0.01):
    def calc_bias_init(W, p=0.01):
        # W is a Tensor of shape (out_features, in_features)
        z_p = norm.ppf(p)
        row_norms = torch.linalg.norm(W, dim=1)  # L2 norm per row
        b = z_p * row_norms
        return b + math.exp(0.1)

    def custom_weights_init(m):
        in_dim = m.weight.size(-1)
        nn.init.uniform_(m.weight, a=-1/math.sqrt(in_dim), 
                         b=1/math.sqrt(in_dim), generator=None)
        if m.bias is not None:
            with torch.no_grad():
                m.bias.copy_(calc_bias_init(m.weight, p))

    return custom_weights_init

class Transcoder(nn.Module):
    def __init__(self, input_size, out_size, n_feats, bias=True,
                  threshhold=0.1, bandwidth=2):
        super(Transcoder, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.n_feats = n_feats
        self.input_to_features = nn.Linear(input_size, n_feats, bias=bias)
        self.features_to_outputs = nn.Linear(n_feats, out_size, bias=bias)
        self.act = JumpReLU(torch.tensor(threshhold), bandwidth)
    
    def forward(self, x):
        pre_feats = self.input_to_features(x)
        feats = self.act(pre_feats)
        replace_out = self.features_to_outputs(feats)
        return replace_out, feats, pre_feats