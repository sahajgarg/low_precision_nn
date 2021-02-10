import torch
import torch.nn as nn

from observers.observer_base import ObserverBase


class MinMaxObserver(ObserverBase):
    def __init__(self, bitwidth, symmetric, dynamic, batch_avg,
            train_bitwidth=False, per_channel_bitwidth=False, train_qminmax=False, 
            train_via_scale=False, round_bitwidth=False, discrete_bitwidth=False, stochastic_bitwidth=False, 
            shape=None, axis=0):
        super().__init__(bitwidth, symmetric, dynamic, batch_avg,
                train_bitwidth, per_channel_bitwidth, train_via_scale,
                round_bitwidth, discrete_bitwidth, stochastic_bitwidth, shape, axis)

        self.train_qminmax = train_qminmax
        self.per_channel_bitwidth = per_channel_bitwidth
        if per_channel_bitwidth and shape:
            param_shape = [1 for _ in shape]
            param_shape[axis] = shape[axis]
        else:
            param_shape = [1]

        # Min and max val will be updated during calibration. If train_qminmax is true, then during training, 
        # these parameters will be learned after initialization from calibration. 
        self.max_val = nn.Parameter(torch.ones(param_shape).float() * -float('inf'), requires_grad=train_qminmax)
        self.min_val = nn.Parameter(torch.ones(param_shape).float() * float('inf'), requires_grad=train_qminmax and not self.symmetric)

    def forward(self, x_in):
        if not self.calibrated or self.dynamic:
            x = x_in.detach()
            if self.batch_avg:
                x = x.view(x.shape[0], -1)
                min_val = torch.mean(torch.min(x, dim=-1)[0])
                max_val = torch.mean(torch.max(x, dim=-1)[0])
            else:
                min_val = torch.min(x)
                max_val = torch.max(x)

            if not self.dynamic:
                min_val = torch.min(self.min_val.data, min_val)
                max_val = torch.max(self.max_val.data, max_val)

            self.min_val.data = min_val
            self.max_val.data = max_val

        return x_in

    def finalize_calibration(self):
        if self.train_via_scale:
            scale, _, _, _ = self.calculate_qparams()
            if self.log_scale.numel() == 1 and scale.numel() > 1:
                self.log_scale.data = torch.log(scale[0])
            else:
                self.log_scale.data = torch.log(scale)

        super().finalize_calibration()

    def symmetric_compression(self):
        symmetric_comp = 2. * torch.max(self.max_val.data, -self.min_val.data) / (self.max_val.data - self.min_val.data)
        return symmetric_comp

    def calculate_bitwidth_from_scale(self):
        return torch.log2(torch.abs(self.max_val -
                            self.min_val) / torch.exp(self.log_scale) + 1.)

    def calculate_qparams(self):
        if self.symmetric and not self.calibrated:
            self.max_val.data = torch.max(self.max_val.data, -self.min_val.data)
        if self.symmetric:
            self.min_val.data = -self.max_val.data

        return self._calculate_qparams(self.min_val, self.max_val)


class EMAMinMaxObserver(MinMaxObserver):
    def __init__(self, bitwidth, symmetric, dynamic, batch_avg, gamma=0.01,
            train_bitwidth=False, per_channel_bitwidth=False, train_qminmax=False, 
            train_via_scale=False, round_bitwidth=False, discrete_bitwidth=False, stochastic_bitwidth=False, 
            shape=None, axis=0):
        super().__init__(bitwidth, symmetric, dynamic, batch_avg,
                train_bitwidth, per_channel_bitwidth, train_qminmax, 
                train_via_scale, round_bitwidth, discrete_bitwidth, stochastic_bitwidth, 
                shape, axis)
        self.gamma = gamma

    def forward(self, x_in):
        if not self.calibrated:
            x = x_in.detach()
            if self.batch_avg:
                x = x.view(x.shape[0], -1)
                min_val = torch.mean(torch.min(x, dim=-1)[0], dim=0)
                max_val = torch.mean(torch.max(x, dim=-1)[0], dim=0)
            else:
                min_val = torch.min(x)
                max_val = torch.max(x)

            if not torch.isinf(self.min_val).any():
                min_val = (1 - self.gamma) * self.min_val.data + self.gamma * min_val
                max_val = (1 - self.gamma) * self.max_val.data + self.gamma * max_val

            self.min_val.data = min_val
            self.max_val.data = max_val

        return x_in
