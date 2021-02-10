import torch.nn as nn
import torch

class QuantStub(nn.Module):
    def __init__(self, qconfig):
        super().__init__()
        self.set_qconfig(qconfig)

    def forward(self, input):
        return self.activation_quantizer(input)

    def set_qconfig(self, qconfig):
        self.activation_quantizer = qconfig['activation']()

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        assert type(mod) == torch.quantization.QuantStub
        stub = cls(qconfig)
        return stub
