import torch
import torch.nn as nn
from quantizers import RoundSTE

# Additive Gaussian noise for thermal, shot, and weight noise
class Gaussian(nn.Module):
    def __init__(self, emac, noise_type, shape, record_snr, record_snr_cascading, quantize_energy):
        super().__init__()
        self.log_emac = nn.Parameter(torch.log(torch.ones(shape) * emac))
        assert noise_type in ["shot", "thermal", "weight"]
        self.noise_type = noise_type
        self.total_macs = None
        self.scale_factor = None
        self.add_noise = True
        self.quantize_energy = quantize_energy

        self.record_snr = record_snr
        self.record_snr_cascading = record_snr_cascading
        self.recording = False
        self.recording_clean = False
        if record_snr or record_snr_cascading:
            self.n = 0
            self.stat_names = ["Sig Mean", "Sig Var", "Noise Mean", "Noise Var"]
            if record_snr_cascading:
                self.stat_names.extend(["Cascaded Noise Mean", "Cascaded Noise Var"])

            self.stats = None
            self.clean = None

    # Shot noise, assuming dot product signal, is given by N(0, 1/sqrt(n_mac)) * ||W_i|| ||x|| / sqrt(D)
    # The scale factor here should absorb ||W_i|| ||x|| / sqrt(D), assuming there is no dynamic quantization of 
    # weight tensors, which doesn't really make much sense anyways. Note that the x is different from the 
    # input to forward, since x is the input to the layer, not the pre-activation.

    # Thermal noise, on the other hand, has a fixed scale relative to the fixed point range of the outputs.
    # Thus, the scale_factor will incorporate the quantizer scale factors and bit precisions to rescale 
    # the noise appropriately.
    def forward(self, x):
        if self.recording_clean:
            self.clean = x.detach()

        if not self.add_noise or self.recording_clean:
            return x

        emac = torch.exp(self.log_emac)
        if self.quantize_energy:
            emac = RoundSTE.apply(emac)

        if self.noise_type == "shot":
            sd = 1. / torch.sqrt(emac)
        elif self.noise_type == "thermal":
            sd = 1. / torch.sqrt(emac) / 100. # sigma_t = 0.01
        elif self.noise_type == "weight":
            sd = 1. / torch.sqrt(emac * 100.) # sigma_w = 0.1

        noise = torch.randn_like(x) * sd * self.scale_factor

        # Scale factor reset every forward pass since params may change.
        self.scale_factor = None  
        if (self.record_snr or self.record_snr_cascading) and self.recording:
            self.record(x, noise)

        return x + noise

    def update_stat(self, mean_key, var_key, values):
        batch_size = values.shape[0]
        tot = self.n + batch_size

        # Note: we update var first since we need to use the old value of the mean in the calc  
        self.stats[var_key] = batch_size / tot * torch.var(values, 0) + self.n / tot * self.stats[var_key] + \
                       (batch_size * self.n) / (tot ** 2) * (torch.mean(values, 0) - self.stats[mean_key]) ** 2
        self.stats[mean_key] = batch_size / tot * torch.mean(values, 0) + self.n / tot * self.stats[mean_key]


    # NOTE: all stats recorded before the activation function.
    def record(self, x_in, noise):
        x = x_in.detach().flatten()
        a_noise = noise.detach().flatten()

        if self.stats is None:
            stats_base = torch.zeros_like(x.mean(0))
            self.stats = {stat: stats_base.clone() for stat in self.stat_names}

        if self.record_snr: 
            self.update_stat("Sig Mean", "Sig Var", x)
            self.update_stat("Noise Mean", "Noise Var", a_noise)
        else:
            self.update_stat("Sig Mean", "Sig Var", self.clean.flatten())
            self.update_stat("Noise Mean", "Noise Var", a_noise)
            self.update_stat("Cascaded Noise Mean", "Cascaded Noise Var", x + a_noise - self.clean.flatten())
            self.clean = None

        self.n += x.shape[0]
