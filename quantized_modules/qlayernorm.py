import torch.nn as nn
import torch.nn.functional as F


class QLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, qconfig={}):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.set_qconfig(qconfig)

    def set_qconfig(self, qconfig):
        self.activation_quantizer = qconfig['activation']()
        # Weights are not quantized for QLayerNorm because they are relatively inexpensive 
        # and quantizing weights substantially reduces accuracy.

    def forward(self, input):
        return self.activation_quantizer(F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps))

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        assert type(mod) == nn.LayerNorm
        layernorm = cls(mod.normalized_shape, mod.eps, mod.elementwise_affine, qconfig)
        layernorm.weight = mod.weight
        layernorm.bias = mod.bias
        return layernorm
