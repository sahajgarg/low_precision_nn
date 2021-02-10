import torch
import torch.nn as nn
import numpy as np

from observers.observer_base import ObserverBase


class EMAPercentileObserver(ObserverBase):
    def __init__(self, bitwidth, symmetric, dynamic, batch_avg, gamma=0.01,
            train_bitwidth=False, per_channel_bitwidth=False, train_qminmax=False, 
            train_via_scale=False, round_bitwidth=False, discrete_bitwidth=False, stochastic_bitwidth=False, 
            shape=None, axis=0, percentile=99):
        super().__init__(bitwidth, symmetric, dynamic, batch_avg,
                train_bitwidth, per_channel_bitwidth, train_via_scale,
                round_bitwidth, discrete_bitwidth, stochastic_bitwidth, shape, axis)

        self.gamma = gamma
        self.percentile = percentile
        assert not (train_qminmax or train_via_scale or batch_avg), "Disabled settings for percentile observer"

        self.per_channel_bitwidth = per_channel_bitwidth
        if per_channel_bitwidth and shape:
            param_shape = [1 for _ in shape]
            param_shape[axis] = shape[axis]
        else:
            param_shape = [1]

        assert not train_qminmax
        self.max_val = nn.Parameter(torch.ones(param_shape).float() * -float('inf'), requires_grad=train_qminmax)
        self.min_val = nn.Parameter(torch.ones(param_shape).float() * float('inf'), requires_grad=train_qminmax and not self.symmetric)

    def forward(self, x_in):
        if not self.calibrated or self.dynamic:
            x = x_in.detach()
            quantiles = np.percentile(x.flatten().detach().cpu().numpy(), [100. - self.percentile, self.percentile])
            min_val = torch.tensor(quantiles[0]).to(x.device)
            max_val = torch.tensor(quantiles[1]).to(x.device)

            # use batch values if dynamic, or EMA if not dynamic
            if not self.dynamic:
                min_val = torch.min(self.min_val.data, min_val)
                max_val = torch.max(self.max_val.data, max_val)
            elif not torch.isinf(self.min_val).any():
                min_val = (1 - self.gamma) * self.min_val.data + self.gamma * min_val
                max_val = (1 - self.gamma) * self.max_val.data + self.gamma * max_val

            self.min_val.data = min_val
            self.max_val.data = max_val

        return x_in

    def calculate_qparams(self):
        if self.symmetric and not self.calibrated:
            self.max_val.data = torch.max(self.max_val.data, -self.min_val.data)
        if self.symmetric:
            self.min_val.data = -self.max_val.data

        return self._calculate_qparams(self.min_val, self.max_val)


class PercentileObserver(ObserverBase):
    def __init__(self, bitwidth, symmetric, dynamic, batch_avg,
            train_bitwidth=False, per_channel_bitwidth=False, train_qminmax=False, 
            train_via_scale=False, round_bitwidth=False, discrete_bitwidth=False, stochastic_bitwidth=False, 
            shape=None, axis=0, percentile=99):
        super().__init__(bitwidth, symmetric, dynamic, batch_avg,
                train_bitwidth, per_channel_bitwidth, train_via_scale,
                round_bitwidth, discrete_bitwidth, stochastic_bitwidth, shape, axis)

        self.percentile = percentile
        assert not (train_qminmax or train_via_scale or batch_avg), "Disabled settings for percentile observer"

        self.per_channel_bitwidth = per_channel_bitwidth
        if per_channel_bitwidth and shape:
            param_shape = [1 for _ in shape]
            param_shape[axis] = shape[axis]
        else:
            param_shape = [1]

        self.values = None

        assert not train_qminmax
        self.max_val = nn.Parameter(torch.ones(param_shape).float() * -float('inf'), requires_grad=train_qminmax)
        self.min_val = nn.Parameter(torch.ones(param_shape).float() * float('inf'), requires_grad=train_qminmax and not self.symmetric)

    def forward(self, x_in):
        if not self.calibrated or self.dynamic:
            x = x_in.detach().flatten().cpu()
            if self.dynamic:
                self.values = x
            else:
                self.values = torch.cat([self.values, x]) if self.values is not None else x

            quantiles = np.percentile(self.values.numpy(), [100. - self.percentile, self.percentile])
            self.min_val.data = torch.tensor(quantiles[0]).to(self.max_val.device)
            self.max_val.data = torch.tensor(quantiles[1]).to(self.max_val.device)

        return x_in

    def finalize_calibration(self):
        # Used when not all operations are quantized, so the observer is not called.
        if self.values is None:
            super().finalize_calibration()
            return

        quantiles = np.percentile(self.values.numpy(), [100. - self.percentile, self.percentile])
        std = np.std(self.values.numpy())
        range_shrinkage = (self.max_val - self.min_val) / (quantiles[1] - quantiles[0])
        # print(f"Normal:\t\t{self.min_val.item():0.2f}\t{self.max_val.item():0.2f}\t{quantiles[0].item():0.2f}\t{quantiles[1].item():0.2f}\t" + 
        #         f"{range_shrinkage.item():0.2f}\t" + 
        #         f"{((self.max_val - self.min_val) / std).item():0.2f}")

        self.min_val.data = torch.tensor(quantiles[0]).to(self.max_val.device)
        self.max_val.data = torch.tensor(quantiles[1]).to(self.max_val.device)
        del self.values
        self.values = None
        super().finalize_calibration()

    def calculate_qparams(self):
        if self.symmetric and not self.calibrated:
            self.max_val.data = torch.max(self.max_val.data, -self.min_val.data)
        if self.symmetric:
            self.min_val.data = -self.max_val.data

        return self._calculate_qparams(self.min_val, self.max_val)
