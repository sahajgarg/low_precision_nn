import torch.nn as nn
import torch.nn.functional as F

class QAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, qconfig):
        super().__init__(output_size)
        self.set_qconfig(qconfig)

    def forward(self, input):
        return self.activation_quantizer(F.adaptive_avg_pool2d(input, self.output_size))

    def set_qconfig(self, qconfig):
        self.quant_gemm_only = qconfig["quant_gemm_only"]
        # NOTE: quant_relu_only is ignored because it was meant to correspond
        # to quantizing the inputs to each layer. Avgpool is frequently
        # excluded, and hence it is equivalent to replacing ReLUs with
        # quantization, but for acceleration benefits, the outputs of avgpool
        # must also be quantized unless they are postprocessed with a relu
        # before input to the next layer.
        self.quant_relu_only = qconfig["quant_relu_only"]
        if self.quant_gemm_only:
            self.activation_quantizer = nn.Identity()
        else:
            self.activation_quantizer = qconfig['activation']()

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        assert type(mod) == nn.AdaptiveAvgPool2d
        avgpool = cls(mod.output_size, qconfig)
        return avgpool


class QAvgPool2d(nn.AvgPool2d):
    def __init__(self, qconfig, kernel_size, stride = None, padding = 0,
                 ceil_mode = False, count_include_pad = True, divisor_override = None):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        self.set_qconfig(qconfig)

    def forward(self, input):
        return self.activation_quantizer(F.avg_pool2d(input, self.kernel_size,
            self.stride, self.padding, self.ceil_mode, self.count_include_pad,
            self.divisor_override))

    def set_qconfig(self, qconfig):
        self.quant_gemm_only = qconfig["quant_gemm_only"]
        self.quant_relu_only = qconfig["quant_relu_only"]
        if self.quant_gemm_only:
            self.activation_quantizer = torch.nn.Identity()
        else:
            self.activation_quantizer = qconfig['activation']()

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        assert type(mod) == nn.AvgPool2d
        avgpool = cls(qconfig, mod.kernel_size, mod.stride, mod.padding, mod.ceil_mode,
                mod.count_include_pad, mod.divisor_override)
        return avgpool
