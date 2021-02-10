from __future__ import absolute_import, division, print_function, unicode_literals

from math import sqrt
from noise_models import *
from observers import ChannelMinMaxObserver

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic
import torch.nn.qat as nnqat
from torch.nn import init
from torch.nn.modules.utils import _pair


class _QConvBnNd(nn.modules.conv._ConvNd):
    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups,
                 # bias: None, only support Conv with no bias
                 padding_mode,
                 # BatchNormNd args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, False, padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.eps = eps
        self.momentum = momentum
        self.freeze_bn = freeze_bn if self.training else True
        self.finalized = False
        self.num_features = out_channels
        self.gamma = nn.Parameter(torch.Tensor(out_channels))
        self.beta = nn.Parameter(torch.Tensor(out_channels))
        self.affine = True
        self.track_running_stats = True
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.reset_bn_parameters()
        self.set_qconfig(qconfig)
        self.idx = None
        self.param_added = False

    def set_qconfig(self, qconfig):
        self.activation_quantizer = qconfig['activation'](shape=[1, self.in_channels, 1, 1], axis=1)
        self.weight_quantizer = qconfig['weight'](shape=self.weight.shape, axis=0)
        self.quant_gemm_only = qconfig['quant_gemm_only']
        self.quant_relu_only = qconfig['quant_relu_only']
        self.act_noise = qconfig['act_noise']
        self.weight_noise = qconfig['weight_noise']
        if self.quant_gemm_only or self.quant_relu_only:
            assert isinstance(self.act_noise, nn.Identity) and \
                    isinstance(self.weight_noise, nn.Identity), \
                    "noise not implemented for quant_gemm_only"

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_bn_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    def reset_parameters(self):
        super(_QConvBnNd, self).reset_parameters()
        # A hack to avoid resetting on undefined parameters
        if hasattr(self, 'gamma'):
            self.reset_bn_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        return self

    def finalize_fusion(self):
        assert self.freeze_bn
        self.finalized = True
        running_std = torch.sqrt(self.running_var + self.eps)
        scale_factor = self.gamma / running_std
        scaled_weight = self.weight * scale_factor.reshape([-1, 1, 1, 1])

        self.weight = nn.Parameter(scaled_weight)
        self.offset = nn.Parameter((self.beta - self.gamma * self.running_mean /
                                    running_std).reshape([1, -1, 1, 1]))
        return self

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

    def _forward(self, input):
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and not self.freeze_bn and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # we use running statistics from the previous batch, so this is an
        # approximation of the approach mentioned in the whitepaper, but we only
        # need to do one convolution in this case instead of two
        running_std = torch.sqrt(self.running_var + self.eps)
        scale_factor = self.gamma / running_std
        scaled_weight = self.weight * scale_factor.reshape([-1, 1, 1, 1])
        weight_to_use = self.weight if self.finalized else scaled_weight
        weight_to_use = self.ecaq_scaling(weight_to_use)
        qweight = self.weight_quantizer(weight_to_use)
        qweight = self.ecaq_scaling(qweight, revert=True)
        self.set_weight_noise_scale_factor()
        self.qweight = self.weight_noise(qweight)
        conv = self._conv_forward(input, self.qweight)

        if not self.finalized:
            # recovering original conv to get original batch_mean and batch_var
            conv_orig = conv / scale_factor.reshape([1, -1, 1, 1])
            batch_mean = torch.mean(conv_orig, dim=[0, 2, 3])
            batch_var = torch.var(conv_orig, dim=[0, 2, 3], unbiased=False)
            n = float(conv_orig.numel() / conv_orig.size()[1])
            unbiased_batch_var = batch_var * (n / (n - 1))
            batch_rstd = torch.ones_like(batch_var, memory_format=torch.contiguous_format) / torch.sqrt(
                batch_var + self.eps)

            rescale_factor = running_std * batch_rstd
            conv = conv * rescale_factor.reshape([1, -1, 1, 1])
            conv = conv + (self.beta - self.gamma * batch_mean * batch_rstd).reshape([1, -1, 1, 1])

            self.running_mean = exponential_average_factor * batch_mean.detach() + \
                                (1 - exponential_average_factor) * self.running_mean
            self.running_var = exponential_average_factor * unbiased_batch_var.detach() + \
                               (1 - exponential_average_factor) * self.running_var
        else:
            conv = conv + self.offset
        return conv

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_QConvBnNd, self).extra_repr()

    def set_weight_noise_scale_factor(self):
        if not isinstance(self.weight_noise, nn.Identity):
            qparam_scaling = self.weight_quantizer.observer.calculate_qparams()[0] 
            bit_precision_scaling = (round(2 ** self.weight_quantizer.observer.bitwidth) - 1) 
            noise_scale = qparam_scaling * bit_precision_scaling
            self.weight_noise.scale_factor = noise_scale

    def set_act_noise_scale_factor(self, input):
        if not isinstance(self.act_noise, nn.Identity):
            dimension = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
            if self.act_noise.noise_type == "thermal":
                # infer the scale for the input to this layer. 
                # inferred because it is challenging to trace the graph to find
                # which outputs of which layer are inputs to the next
                if not self.activation_quantizer.observer.calibrated:
                    self.inferred_input_scale = torch.abs(input)[torch.abs(input) > 1e-20].min()
                    max_round_error = torch.abs(input / self.inferred_input_scale - torch.round(input / self.inferred_input_scale)).max()
                    int_range = (input.max() - input.min()) / self.inferred_input_scale
                    assert max_round_error < 1e-4 and int_range <= 2 ** 8,  f"{max_round_error} {int_range}"
                qparam_scaling = self.weight_quantizer.observer.calculate_qparams()[0].transpose(0, 1) * \
                        self.inferred_input_scale
                bit_precision_scaling = (round(2 ** self.weight_quantizer.observer.bitwidth) - 1) * \
                    (round(2 ** self.activation_quantizer.observer.bitwidth) - 1)
                noise_scale = qparam_scaling * bit_precision_scaling * sqrt(dimension)
            if self.act_noise.noise_type == "shot":
                assert self.padding_mode == 'zeros'
                assert self.bias is None
                assert self.groups == 1 or self.groups == self.out_channels, \
                        "Only expected to work correctly for normal or fully depthwise convs"
                # Sometimes the norms are negative due to floating point rounding errors. Weird.
                input_norms = torch.sqrt(F.relu(F.conv2d(input.detach() ** 2., torch.ones_like(self.qweight[:self.groups].detach()), 
                    None, self.stride, self.padding,
                    self.dilation, self.groups)))
                weight_norms = torch.sqrt(F.relu(F.conv2d(torch.ones_like(input[:1].detach()), self.qweight.detach() ** 2,
                    None, self.stride, self.padding, self.dilation, self.groups)))

                noise_scale = input_norms * weight_norms / sqrt(dimension)
            self.act_noise.scale_factor = noise_scale

    def set_total_macs(self, output):
        self.act_noise.total_macs = (self.in_channels / self.groups) * \
                self.kernel_size[0] * self.kernel_size[1] * torch.prod(torch.tensor(output.shape[1:]))
        self.act_noise.num_neurons = torch.prod(torch.tensor(output.shape[1:]))
        self.weight_noise.total_macs = (self.in_channels / self.groups) * \
                self.kernel_size[0] * self.kernel_size[1] * torch.prod(torch.tensor(output.shape[1:]))

    def forward(self, input):
        if self.quant_gemm_only:
            output = self._forward(self.activation_quantizer(input))
        else:
            output = self._forward(input)
            self.set_act_noise_scale_factor(input)
            if not self.quant_relu_only: 
                output = self.activation_quantizer(self.act_noise(output))
        self.set_total_macs(output)
        return output

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
                                               cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups,
                         conv.padding_mode,
                         bn.eps, bn.momentum,
                         False,
                         qconfig)
        assert qat_convbn.bias is None, 'QAT ConvBn should not have bias'
        qat_convbn.weight = conv.weight
        qat_convbn.gamma = bn.weight
        qat_convbn.beta = bn.bias
        qat_convbn.running_mean = bn.running_mean
        qat_convbn.running_var = bn.running_var
        qat_convbn.num_batches_tracked = bn.num_batches_tracked
        if not qconfig['train']:
            qat_convbn.freeze_bn_stats()
            qat_convbn.finalize_fusion()

        return qat_convbn


class QConvBn2D(_QConvBnNd, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        activation_post_process: fake quant module for output activation
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = torch.nn.intrinsic.ConvBn2d

    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 # bias: None, only support Conv with no bias
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _QConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                            padding, dilation, False, _pair(0), groups, padding_mode,
                            eps, momentum, freeze_bn, qconfig)


class QConvBnReLU2d(QConvBn2D):
    r"""
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = torch.nn.intrinsic.ConvBnReLU2d

    def __init__(self,
                 # Conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 # bias: None, only support Conv with no bias
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        super(QConvBnReLU2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups,
                                            padding_mode, eps, momentum,
                                            freeze_bn,
                                            qconfig)

    def forward(self, input):
        if self.quant_gemm_only:
            output = F.relu(QConvBn2D._forward(self, self.activation_quantizer(input)))
        else:
            output = QConvBn2D._forward(self, input)
            self.set_act_noise_scale_factor(input)
            output = self.activation_quantizer(self.act_noise(output), fused_relu=True)
        self.set_total_macs(output)
        return output

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        return super(QConvBnReLU2d, cls).from_float(mod, qconfig, param_list)


class QConvReLU2d(nnqat.Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for both output activation and weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        activation_quantizer: fake quant module for output activation
        weight_quantizer: fake quant module for weight

    """
    _FLOAT_MODULE = torch.nn.intrinsic.ConvReLU2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 qconfig=None):
        super(QConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation,
                                          groups=groups, bias=bias, padding_mode=padding_mode,
                                          qconfig=qconfig)
        assert qconfig, 'qconfig must be provided for QAT module'
        raise NotImplementedError
        self.qconfig = qconfig
        self.activation_quantizer = self.qconfig['activation']
        self.weight_quantizer = self.qconfig['weight']
        self.idx = None
        self.param_added = False

    def forward(self, input):
        weight = self.ecaq_scaling(self.weight)
        qweight = self.weight_quantizer(weight)
        qweight = self.ecaq_scaling(qweight, revert=True)
        if self.quant_gemm_only:
            output = F.relu(self._conv_forward(self.activation_quantizer(input), self.qweight))
        else: 
            self.set_weight_noise_scale_factor()
            self.qweight = self.weight_noise(qweight)
            output = self._conv_forward(input, self.qweight)
            self.set_act_noise_scale_factor(input)
            output = self.activation_quantizer(self.act_noise(output), fused_relu=True)
        self.set_total_macs(output)
        return output

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        conv = super(QConvReLU2d, cls).from_float(mod, qconfig)
        return conv


def update_bn_stats(mod):
    if type(mod) in set([QConvBnReLU2d, QConvBn2D]):
        mod.update_bn_stats()


def freeze_bn_stats(mod):
    if type(mod) in set([QConvBnReLU2d, QConvBn2D]):
        mod.freeze_bn_stats()
