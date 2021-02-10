import torch

from observers.observer_base import ObserverBase

# ACIQ observer from https://arxiv.org/abs/1810.05723
class ACIQObserver(ObserverBase):
    # FULL_LOOKUP = (1.683, 2.8307, 3.8972, 5.0286, 6.2048, 7.31313, 8.6456, 9.8968)
    FUSED_LOOKUP = (2.8307, 3.8972, 5.0286, 6.2048, 7.31313, 8.6456, 9.8968, 11.1627)

    # FUSED_LOOKUP = [1.71, 2.15, 2.55, 2.93, 3.28, 3.61, 3.92, 4.2]

    def __init__(self, bitwidth, symmetric, dynamic, batch_avg, relu=True):
        super().__init__(bitwidth, symmetric, dynamic, batch_avg)
        self.lookup = self.FUSED_LOOKUP if relu else self.FULL_LOOKUP
        # Seems better with batch_avg = false
        assert not batch_avg
        self.batch_avg = batch_avg 
        if self.dynamic:
            self.min_val = 0
            self.max_val = 0
        else:
            self.mean = 0
            self.count = 0
            self.squared = 0

    def forward(self, x_in):
        x = x_in.detach()
        if self.dynamic:
            if self.batch_avg:
                x = x.view(x.shape[0], -1)
                mean = torch.mean(torch.mean(x, dim=-1))
                std = torch.mean(torch.std(x, dim=-1))
                scale = std / (2 ** 0.5)
            else:
                scale = torch.mean(torch.abs(x - x.mean()))  
                mean = torch.mean(x)
            alpha = self.lookup[self.bitwidth - 1] * scale
            self.min_val = mean - alpha
            self.max_val = mean + alpha
        else:
            if self.batch_avg:
                x = x.view(x.shape[0], -1)
                self.count += x.shape[1]
                delta = x - self.mean
                self.mean += torch.sum(delta / self.count, dim=-1, keepdim=True)
                delta2 = x - self.mean
                self.squared += torch.sum(delta * delta2, dim=-1, keepdim=True)
            else:
                self.count += x.numel()
                delta = x - self.mean
                self.mean += torch.sum(delta / self.count)
                delta2 = x - self.mean
                self.squared += torch.sum(delta * delta2)
        return x_in

    def calculate_qparams(self):
        if self.dynamic:
            return self._calculate_qparams(self.min_val, self.max_val)
        else:
            variance = torch.mean(self.squared / (self.count - 1))
            scale = torch.sqrt(variance / 2)
            alpha = self.lookup[self.bitwidth - 1] * scale
            return self._calculate_qparams(torch.mean(self.mean) - alpha, torch.mean(self.mean) + alpha)
