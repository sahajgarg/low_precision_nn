import torch
import torch.nn as nn
from quantizers import RoundSTE, DiscreteSTE, StochasticSTE

class ObserverBase(nn.Module):
    def __init__(self, bitwidth, symmetric, dynamic, batch_avg, 
            train_bitwidth=False, per_channel_bitwidth=False,
            train_via_scale=False, round_bitwidth=False, discrete_bitwidth=False, stochastic_bitwidth=False, 
            shape=None, axis=0):
        super().__init__()
        self.symmetric = symmetric
        self.train_bitwidth = train_bitwidth 
        self.train_via_scale = train_via_scale
        self.bitwidth_per_channel = per_channel_bitwidth
        self.round_bitwidth = round_bitwidth
        self.discrete_bitwidth = discrete_bitwidth
        self.stochastic_bitwidth = stochastic_bitwidth
        if train_bitwidth:
            if per_channel_bitwidth and shape:
                param_shape = [1 for _ in shape]
                param_shape[axis] = shape[axis]
            else:
                param_shape = [1]

            if train_via_scale:
                self.log_scale = nn.Parameter(torch.ones(param_shape).float())
            else:
                self.log_bitwidth = nn.Parameter(torch.log2(torch.ones(param_shape).float() * bitwidth))
        self.bitwidth = bitwidth
        self.batch_avg = batch_avg
        self.dynamic = dynamic
        self.calibrated = False

    def finalize_calibration(self):
        self.calibrated = True

    def calculate_qparams(self):
        raise NotImplementedError

    def calculate_bitwidth_from_scale(self):
        raise NotImplementedError

    def _calculate_qparams(self, min_val, max_val):
        if self.train_bitwidth and self.train_via_scale and self.calibrated: 
            bitwidth = self.calculate_bitwidth_from_scale()
        elif self.train_bitwidth and not (self.train_via_scale and not self.calibrated):
            bitwidth = 2. ** self.log_bitwidth
        else:
            bitwidth = self.bitwidth 

        if self.train_bitwidth and self.round_bitwidth and not (self.train_via_scale and not self.calibrated): 
            bitwidth = RoundSTE.apply(bitwidth)
        elif self.discrete_bitwidth:
            bitwidth = DiscreteSTE.apply(bitwidth)
        elif self.stochastic_bitwidth:
            bitwidth = StochasticSTE.apply(bitwidth)

        # Guarantee an integer number of bins, and record the corresponding bitwidth used.
        bins = 2. ** (bitwidth) - 1
        if isinstance(bins, float):
            bins = float(round(bins))
            self.bitwidth_for_penalty = torch.log2(torch.tensor(bins)).to(min_val.device)
        else:
            bins = RoundSTE.apply(bins)
            self.bitwidth_for_penalty = torch.where(bins > 0, torch.log2(bins + 1.), torch.zeros_like(bins))

        if not self.symmetric:
            if isinstance(bins, float):
                scale = (max_val - min_val) / (bins)
            else:
                # prune channels if they are set to zero bitwidth by setting scale to be very large
                scale = torch.where(bins > 1e-5, (max_val - min_val) / (bins), torch.ones_like(bins) * 1e10)
            scale = torch.where(scale < 1e-20, torch.ones_like(scale), scale)
            zero_point = RoundSTE.apply(-min_val / scale)
        else:
            scale = 2 * max_val / (bins)
            scale = torch.where(scale < 1e-20, torch.ones_like(scale), scale)
            zero_point = 0.

        return scale, zero_point, min_val, max_val

    def forward(self, x):
        raise NotImplementedError
