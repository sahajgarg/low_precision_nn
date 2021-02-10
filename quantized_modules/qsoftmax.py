import torch.nn as nn
import torch.nn.functional as F

class QSoftmax(nn.Softmax):
    def __init__(self, dim=None, qconfig={}):
        super().__init__(dim)
        self.set_qconfig(qconfig)

    def set_qconfig(self, qconfig):
        self.activation_quantizer = qconfig['activation']()

    def forward(self, input):
        return self.activation_quantizer(F.softmax(
            input, self.dim, _stacklevel=5))

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        assert type(mod) == nn.Softmax
        softmax = cls(mod.dim, qconfig)
        return softmax
