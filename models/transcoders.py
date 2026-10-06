from typing import Any
import math 

from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rnn import myModule

#code adapted from https://github.com/safety-research/circuit-tracer/blob/main/
def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)

class jumprelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, log_threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        # ``threshold`` is stored in log-space so it remains positive.  Keep
        # the effective value in the autograd context: the forward gate and
        # its straight-through gradient must use the same threshold.
        threshold = torch.exp(log_threshold)
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x * (x > threshold)).to(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        # Banded straight-through estimator for the log-threshold parameter.
        # The factor of threshold accounts for d exp(log_threshold) / d
        # log_threshold.  The old version mixed log- and linear-space values.
        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output,
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

class Transcoder(myModule):
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


@torch.no_grad()
def initialize_paired_dpi(
    transcoder: Transcoder,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    datapoint_scale: float = 0.4,
    activation_density: float = 0.01,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Initialize a supervised transcoder from paired calibration examples.

    Standard DPI initializes an SAE decoder as the encoder transpose, which is
    not possible here because transcoders map input_dim -> output_dim.  This
    paired variant uses the input half of sampled (x, y) pairs for encoder rows
    and their centered targets for decoder columns.  Encoder biases are then
    calibrated on the sampled *raw* inputs to target a chosen initial firing
    density under JumpReLU.
    """
    if not 0.0 <= datapoint_scale <= 1.0:
        raise ValueError("datapoint_scale must be in [0, 1]")
    if not 0.0 < activation_density < 1.0:
        raise ValueError("activation_density must be in (0, 1)")
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("DPI inputs and targets must both be rank-2 tensors")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("DPI inputs and targets must have the same sample count")
    if inputs.shape[1] != transcoder.input_size or targets.shape[1] != transcoder.out_size:
        raise ValueError("DPI calibration tensors do not match transcoder dimensions")
    if inputs.shape[0] < transcoder.n_feats:
        raise ValueError("DPI needs at least one calibration example per feature")

    dtype = transcoder.input_to_features.weight.dtype
    inputs = inputs.to(dtype=dtype)
    targets = targets.to(dtype=dtype)

    input_mean = inputs.mean(dim=0)
    centered_inputs = inputs - input_mean
    # Anthropic's DPI rescales activations so their mean norm is sqrt(d). We
    # apply that calibration only to the sampled encoder directions; model
    # inputs themselves remain in their original units.
    mean_centered_norm = centered_inputs.norm(dim=-1).mean().clamp_min(1e-8)
    input_scale = math.sqrt(transcoder.input_size) / mean_centered_norm
    calibrated_inputs = centered_inputs * input_scale

    selected = torch.randperm(inputs.shape[0], generator=generator)[:transcoder.n_feats]
    encoder_data = calibrated_inputs[selected]
    target_mean = targets.mean(dim=0)
    decoder_data = (targets[selected] - target_mean).T

    random_encoder = torch.empty_like(transcoder.input_to_features.weight)
    random_decoder = torch.empty_like(transcoder.features_to_outputs.weight)
    nn.init.kaiming_uniform_(random_encoder, a=math.sqrt(5))
    nn.init.kaiming_uniform_(random_decoder, a=math.sqrt(5))

    transcoder.input_to_features.weight.copy_(
        datapoint_scale * encoder_data + (1.0 - datapoint_scale) * random_encoder
    )
    transcoder.features_to_outputs.weight.copy_(
        datapoint_scale * decoder_data + (1.0 - datapoint_scale) * random_decoder
    )

    # Match the initial threshold-crossing rate on actual inputs, accounting
    # for non-isotropic data and the encoder-row norms produced by DPI.
    raw_preacts = inputs @ transcoder.input_to_features.weight.T
    threshold = torch.exp(transcoder.act.threshold.detach())
    cutoff = torch.quantile(raw_preacts, 1.0 - activation_density, dim=0)
    transcoder.input_to_features.bias.copy_(threshold - cutoff)
    transcoder.features_to_outputs.bias.copy_(target_mean)

    calibrated_preacts = raw_preacts + transcoder.input_to_features.bias
    observed_density = (calibrated_preacts > threshold).float().mean()
    return {
        "datapoint_scale": float(datapoint_scale),
        "activation_density_target": float(activation_density),
        "activation_density_observed": float(observed_density),
        "calibration_samples": int(inputs.shape[0]),
        "input_centered_mean_norm": float(mean_centered_norm),
        "input_direction_scale": float(input_scale),
    }
