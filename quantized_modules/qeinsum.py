import torch
import torch.nn as nn 

from models.bert import Einsum

class QEinsum(Einsum):
    def __init__(self, qconfig={}):
        super().__init__()
        self.set_qconfig(qconfig)
        self.dim = None

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        assert type(mod) == Einsum
        einsum = cls(qconfig)
        return einsum

    def set_qconfig(self, qconfig):
        self.activation_quantizer = qconfig['activation']()
        self.act_noise = qconfig['act_noise']

    def set_act_noise_scale_factor(self, formula, x, y):
        if not isinstance(self.act_noise, nn.Identity):
            if self.act_noise.noise_type == "thermal":
                raise NotImplementedError
            if self.act_noise.noise_type == "shot":
                x_norms = torch.sqrt(torch.einsum(formula, x ** 2., torch.ones_like(y)))
                y_norms = torch.sqrt(torch.einsum(formula, torch.ones_like(x), y ** 2.))
                if self.dim is None:
                    self.dim = torch.einsum(formula, torch.ones_like(x), torch.ones_like(y)).mean().int()
                    self.act_noise.total_macs = self.dim.item() * torch.prod(torch.tensor(x_norms.shape[1:]))
                
                noise_scale = x_norms * y_norms / torch.sqrt(self.dim.float())
            self.act_noise.scale_factor = noise_scale

    def forward(self, formula, x, y):
        self.set_act_noise_scale_factor(formula, x, y)
        output = self.activation_quantizer(self.act_noise(torch.einsum(formula, x, y)))
        return output
