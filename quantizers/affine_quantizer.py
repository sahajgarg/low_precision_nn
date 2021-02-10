import inspect
import math
from random import randint
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Round, allowing gradients to flow backwards
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# Quantize the input to the nearest bitwidth in the subset {2, 4, 8}
class DiscreteSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        bit_set = torch.tensor([2., 4., 8.]).to(input.device)
        min_diff = 100
        diff = 100
        for bit in bit_set:
            curr_diff = bit-input
            abs_diff = torch.abs(bit-input)
            if abs_diff < min_diff:
                min_diff = abs_diff
                diff = curr_diff
        return input + diff

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# Stochastically round the input
class StochasticSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input) + ((input - torch.floor(input)) > torch.rand(input.shape).to(input.device))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# NOTE: because this quantization does rounding before offset, it guarantees that zero is 
# exactly represented and therefore the zero point is not actually used.
def quantize_affine(input, scale, zero_point, min_val, max_val):
    clamped = torch.max(torch.min(input, max_val), min_val)
    input_quantized = zero_point + RoundSTE.apply(clamped / scale)
    dequantized = (input_quantized - zero_point) * scale
    return dequantized

class AffineQuantizer(nn.Module):
    def __init__(self, observer, act_quant_before_relu=False, bias_correct=False,
                 shape=None, axis=None, noise_only=False):
        super().__init__()
        if axis:
            self.observer = observer(shape=shape, axis=axis)
        else:
            self.observer = observer()

        self.approximate_quantize = False
        self.act_quant_before_relu = act_quant_before_relu
        self.bias_correct = bias_correct
        self.noise_only = noise_only
        self.input_shape = None

    # Correct for bias and variance of quantization, which can then be folded into 
    # quantization parameters. This is only applied for weights. 
    def correct(self, x_in, x_quantized):
        bias_q = x_quantized.view(x_quantized.shape[0], -1).mean(-1)
        bias_orig = x_in.view(x_in.shape[0], -1).mean(-1)
        if x_quantized.shape[-1] == 1:
            var_corr = torch.tensor([1]).to(x_in.device)
        else:
            x_std = torch.std(x_in)
            quantized_std = torch.std(x_quantized)
            var_corr = x_std / quantized_std

        shape = [-1] + [1] * (len(x_in.shape) - 1)
        bias_q = bias_q.view(*shape)
        bias_orig = bias_orig.view(*shape)
        var_corr = var_corr.view(*shape)
        x_quantized = (x_quantized - bias_q) * var_corr + bias_orig
        return x_quantized

    # Perform observer logging and quantization
    def forward(self, x_in, fused_relu=False):
        nonlinearity = F.relu if fused_relu else lambda x: x

        if self.noise_only:
            return nonlinearity(x_in) if fused_relu else x_in 

        if self.input_shape is None:
            self.input_shape = x_in.shape

        if fused_relu and not self.act_quant_before_relu:
            x_in = nonlinearity(x_in)

        self.observer(x_in)
        scale, zero_point, min_val, max_val = self.observer.calculate_qparams()

        if self.approximate_quantize:
            dither = (torch.rand_like(x_in) - 0.5) * scale
            x_quantized = x_in + dither
            x_quantized = torch.max(torch.min(x_quantized, max_val), min_val)
        else:
            x_quantized = quantize_affine(x_in, scale, zero_point, min_val, max_val)

        if self.bias_correct:
            x_quantized = self.correct(x_in, x_quantized)

        if fused_relu and self.act_quant_before_relu:
            x_quantized = nonlinearity(x_quantized)

        return x_quantized
