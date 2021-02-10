import torch

from observers.minmax_observer import MinMaxObserver


class ChannelMinMaxObserver(MinMaxObserver):
    def __init__(self, bitwidth, symmetric, axis=0,
            train_bitwidth=False, per_channel_bitwidth=False, train_qminmax=False, 
            train_via_scale=False, round_bitwidth=False, discrete_bitwidth=False, stochastic_bitwidth=False,
            shape=None):
        super().__init__(bitwidth, symmetric, False, False,
                train_bitwidth, per_channel_bitwidth, train_qminmax, 
                train_via_scale, round_bitwidth, discrete_bitwidth, stochastic_bitwidth, shape, axis=axis)
        self.axis = axis

    def forward(self, x_in):
        if not self.calibrated or self.dynamic:
            x = x_in.detach()
            axes = [dim for dim in range(len(x.shape)) if dim != self.axis]

            if axes == []:
                min_val = x
                max_val = x
            else:
                min_val = torch.min(x, dim=axes[0], keepdim=True).values
                max_val = torch.max(x, dim=axes[0], keepdim=True).values
                for axis in axes[1:]:
                    min_val = torch.min(min_val, dim=axis, keepdim=True).values
                    max_val = torch.max(max_val, dim=axis, keepdim=True).values

            if not self.dynamic:
                min_val = torch.min(self.min_val.data, min_val)
                max_val = torch.max(self.max_val.data, max_val)

            self.min_val.data = min_val 
            self.max_val.data = max_val

        return x_in

    def calculate_bitwidth_from_scale(self):
        if self.per_channel_bitwidth:
            return torch.log2(torch.abs(self.max_val -
                                self.min_val) / torch.exp(self.log_scale) + 1.)
        else:
            return torch.log2(torch.abs(self.max_val[0] -
                                self.min_val[0]) / torch.exp(self.log_scale) + 1.)


class EMAChannelMinMaxObserver(ChannelMinMaxObserver):
    def __init__(self, bitwidth, symmetric, axis=0, gamma=0.01,
            train_bitwidth=False, per_channel_bitwidth=False, train_qminmax=False, 
            train_via_scale=False, round_bitwidth=False, discrete_bitwidth=False, stochastic_bitwidth=False, shape=None):
        super().__init__(bitwidth, symmetric, axis,
                train_bitwidth, per_channel_bitwidth, train_qminmax, 
                train_via_scale, round_bitwidth, discrete_bitwidth, stochastic_bitwidth, shape)
        self.gamma = gamma

    def forward(self, x_in):
        if not self.calibrated:
            x = x_in.detach()
            axes = [dim for dim in range(len(x.shape)) if dim != self.axis]

            if axes == []:
                if torch.isinf(self.min_val).any():
                    min_val = x
                    max_val = x
                else:
                    min_val = (1 - self.gamma) * self.min_val + self.gamma * x
                    max_val = (1 - self.gamma) * self.max_val + self.gamma * x
            else:
                min_val = torch.min(x, dim=axes[0], keepdim=True).values
                max_val = torch.max(x, dim=axes[0], keepdim=True).values
                for axis in axes[1:]:
                    min_val = torch.min(min_val, dim=axis, keepdim=True).values
                    max_val = torch.max(max_val, dim=axis, keepdim=True).values

                if torch.isinf(self.min_val).any():
                    min_val = min_val
                    max_val = max_val
                else:
                    min_val = (1 - self.gamma) * self.min_val + self.gamma * min_val
                    max_val = (1 - self.gamma) * self.max_val + self.gamma * max_val

            self.min_val.data = min_val
            self.max_val.data = max_val

        return x_in
