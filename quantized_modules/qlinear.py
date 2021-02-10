from math import sqrt
from noise_models import *
from observers import ChannelMinMaxObserver

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, qconfig={}):
        super().__init__(in_features, out_features, bias)
        self.set_qconfig(qconfig)
        self.idx = None
        self.param_added = False

    def set_qconfig(self, qconfig):
        self.activation_quantizer = qconfig['activation'](shape=[1, self.in_features], axis=1)
        self.weight_quantizer = qconfig['weight'](shape=self.weight.shape, axis=0)
        self.act_noise = qconfig['act_noise']
        self.weight_noise = qconfig['weight_noise']
        self.quant_gemm_only = qconfig['quant_gemm_only']
        self.quant_relu_only = qconfig['quant_relu_only']
        if self.quant_gemm_only or self.quant_relu_only:
            assert isinstance(self.act_noise, nn.Identity) and \
                    isinstance(self.weight_noise, nn.Identity), \
                    "noise not implemented for quant_gemm_only"
        if isinstance(self.activation_quantizer.observer, ChannelMinMaxObserver):
            assert self.quant_gemm_only, "ECAQ only implemented with precision highway"

    def set_act_noise_scale_factor(self, input):
        if not isinstance(self.act_noise, nn.Identity):
            if self.act_noise.noise_type == "thermal":
                if not self.activation_quantizer.observer.calibrated:
                    # infer the scale for the input to this layer. 
                    # inferred because it is challenging to trace the graph to find
                    # which outputs of which layer are inputs to the next
                    self.inferred_input_scale = torch.abs(input)[torch.abs(input) > 1e-20].min()
                    max_round_error = torch.abs(input / self.inferred_input_scale - torch.round(input / self.inferred_input_scale)).max()
                    int_range = (input.max() - input.min()) / self.inferred_input_scale
                    assert max_round_error < 1e-4 and int_range <= 2 ** 8,  f"{max_round_error} {int_range}"

                qparam_scaling = self.weight_quantizer.observer.calculate_qparams()[0].transpose(0, 1) * \
                    self.inferred_input_scale
                dimension_scaling = sqrt(self.in_features)
                bit_precision_scaling = (round(2 ** self.weight_quantizer.observer.bitwidth) - 1) * \
                    (round(2 ** self.activation_quantizer.observer.bitwidth) - 1)
                noise_scale = qparam_scaling * dimension_scaling * bit_precision_scaling
            elif self.act_noise.noise_type == "shot":
                input_norms = torch.sqrt((input ** 2).sum(-1, keepdim=True))
                weight_norms = torch.sqrt((self.weight_quantizer(self.weight) ** 2).sum(1, keepdim=True)).T
                noise_scale = input_norms * weight_norms / sqrt(self.in_features)

            self.act_noise.scale_factor = noise_scale

    def set_total_macs(self, output):
        self.act_noise.total_macs = self.in_features * torch.prod(torch.tensor(output.shape[1:]))
        self.act_noise.num_neurons = torch.tensor(output.shape[1:])
        self.weight_noise.total_macs = self.in_features * torch.prod(torch.tensor(output.shape[1:]))

    def set_weight_noise_scale_factor(self):
        if not isinstance(self.weight_noise, nn.Identity):
            qparam_scaling = self.weight_quantizer.observer.calculate_qparams()[0]
            bit_precision_scaling = (round(2 ** self.weight_quantizer.observer.bitwidth) - 1) 
            noise_scale = qparam_scaling * bit_precision_scaling
            self.weight_noise.scale_factor = noise_scale

    # If activations are quantized per channel, fold the per-channel quantization parameters
    # into the weights of the next layer, like batch norm fusion. 
    def ecaq_scaling(self, weight, revert=False):
        if isinstance(self.activation_quantizer.observer, ChannelMinMaxObserver):
            rng = self.activation_quantizer.observer.max_val - self.activation_quantizer.observer.min_val
            rng = torch.where(rng <= 1e-10, torch.ones_like(rng), rng)
            if revert:
                return weight / rng
            else:
                return weight * rng
        return weight

    def forward(self, x):
        weight = self.ecaq_scaling(self.weight)
        qweight = self.weight_quantizer(weight)
        qweight = self.ecaq_scaling(qweight, revert=True)
        if self.quant_gemm_only:
            output = F.linear(self.activation_quantizer(x), qweight, self.bias)
        else:   
            self.set_weight_noise_scale_factor()
            qweight = self.weight_noise(qweight)
            output = F.linear(x, qweight, self.bias)
            self.set_act_noise_scale_factor(x)
            if not self.quant_relu_only:
                output = self.activation_quantizer(self.act_noise(output))
        self.set_total_macs(output)
        return output

    @classmethod
    def from_float(cls, mod, qconfig={}, param_list={}):
        assert type(mod) == nn.Linear
        qlinear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, qconfig=qconfig)
        qlinear.weight = mod.weight
        qlinear.bias = mod.bias
        return qlinear


class QLinearReLU(QLinear):
    def __init__(self, in_features, out_features, bias=True, qconfig=None):
        super().__init__(in_features, out_features, bias, qconfig)

    def forward(self, input):
        weight = self.ecaq_scaling(self.weight)
        qweight = self.weight_quantizer(weight)
        qweight = self.ecaq_scaling(qweight, revert=True)
        if self.quant_gemm_only:
            output = F.relu(F.linear(self.activation_quantizer(x), qweight, self.bias))
        else:
            self.set_weight_noise_scale_factor()
            qweight = self.weight_noise(qweight)
            output = F.linear(x, qweight, self.bias)
            self.set_act_noise_scale_factor(x)
            output = self.activation_quantizer(self.act_noise(output), fused_relu=True)
        self.set_total_macs(output)
        return output 

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        return super().from_float(mod, qconfig, param_list)
