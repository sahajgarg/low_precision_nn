import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import torch.nn as nn
import pandas as pd
import torch

from main import MODULE_DICT
from quantized_modules import *
from quantizers import RoundSTE

stats_layer = 0

def get_stats(module, args, override_with_noise_bits):
    global stats_layer 
    stats = {"snr": [], "layers": [], "dims": [], "num_neurons": [], "avg_emac": [], "noise_bits": []}
    if args.record_snr_cascading:
        stats["snr_cascading"] = []

    for name, mod in module.named_children():
        if type(mod) not in MODULE_DICT.values():
            child_stats = get_stats(mod, args, override_with_noise_bits)
            if child_stats is not None:
                for stat_name in stats:
                    stats[stat_name].append(child_stats[stat_name])
        elif not isinstance(mod, FloatFunctional) and not isinstance(mod, QAdaptiveAvgPool2d)\
                and not isinstance(mod, QuantStub) and not isinstance(mod, QAvgPool2d):
            stats_layer += 1
            if stats_layer == 1 and "first" in args.ignore:
                continue
            elif isinstance(mod.activation_quantizer, nn.Identity) and "last" in args.ignore: 
                continue

            if isinstance(mod, nn.Conv2d):
                N = mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
            elif isinstance(mod, nn.Linear):
                N = mod.in_features

            sig_var = mod.act_noise.stats["Sig Var"].cpu().numpy().flatten()
            noise_var = mod.act_noise.stats["Noise Var"].cpu().numpy().flatten()
            sig = np.log10(sig_var)
            noise = np.log10(noise_var)
            num_neurons = mod.act_noise.num_neurons

            if args.noise_type == "weight":
                if args.quantize_energy:
                    avg_emac = torch.mean(RoundSTE.apply(torch.exp(mod.weight_noise.log_emac)))
                else:
                    avg_emac = torch.mean(torch.exp(mod.weight_noise.log_emac))
            else:
                if args.quantize_energy:
                    avg_emac = torch.mean(RoundSTE.apply(torch.exp(mod.act_noise.log_emac)))
                else:
                    avg_emac = torch.mean(torch.exp(mod.act_noise.log_emac))

            if args.noise_type == "thermal":
                qnoise_range = (mod.activation_quantizer.observer.max_val - mod.activation_quantizer.observer.min_val).item()
                qnoise_std = qnoise_range / np.sqrt(12)
                noise_bits = np.log2(np.ceil((qnoise_std / np.sqrt(noise_var) + 1.)))
                stats["noise_bits"].append(noise_bits)
                snr_offset = 10 * (sig - noise) - 20 * np.log10(2. ** noise_bits - 1)
                print(f"SNR: {10 * (sig - noise)}, Noise Bits: {noise_bits}, SNR offset: {snr_offset}")

            if override_with_noise_bits:
                mod.activation_quantizer.observer.bitwidth = (noise_bits)[0] 
            if args.record_snr_cascading:
                noise_var_cascading = mod.act_noise.stats["Cascaded Noise Var"].cpu().numpy().flatten()
                noise_cascading = np.log10(noise_var_cascading)
                stats["snr_cascading"].append(10 * (sig - noise_cascading))

            stats["snr"].append(10 * (sig - noise))
            stats["dims"].append((np.ones_like(sig) * N).astype(int))
            stats["layers"].append((np.ones_like(sig) * stats_layer).astype(int))
            stats["num_neurons"].append((np.ones_like(sig) * num_neurons.item()).astype(int))
            stats["avg_emac"].append((np.ones_like(sig) * avg_emac.item()))

    if stats["snr"]:
        return {stat_name: np.concatenate(stats[stat_name], axis=None) for stat_name in stats}
    return None 


emacs_layer = 0

def get_emac_stats(module, args):
    global emacs_layer 
    stats = {"Layer": [], "Neuron Dimension (D)": [], "Num Neurons (N')": [], "Avg Energy/MAC": []}

    for name, mod in module.named_children():
        if type(mod) not in MODULE_DICT.values():
            child_stats = get_emac_stats(mod, args)
            if child_stats is not None:
                for stat_name in stats:
                    stats[stat_name].append(child_stats[stat_name])
        elif not isinstance(mod, FloatFunctional) and not isinstance(mod, QAdaptiveAvgPool2d)\
                and not isinstance(mod, QuantStub) and not isinstance(mod, QAvgPool2d):
            emacs_layer += 1
            if emacs_layer == 1 and "first" in args.ignore:
                continue
            elif isinstance(mod.activation_quantizer, nn.Identity) and "last" in args.ignore: 
                continue

            if isinstance(mod, nn.Conv2d):
                N = mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
            elif isinstance(mod, nn.Linear):
                N = mod.in_features

            num_neurons = mod.act_noise.num_neurons
            if args.noise_type == "weight":
                if args.quantize_energy:
                    emac = (RoundSTE.apply(torch.exp(mod.weight_noise.log_emac))).detach().cpu().numpy().flatten()
                else:
                    emac = (torch.exp(mod.weight_noise.log_emac)).detach().cpu().numpy().flatten()
            else:
                if args.quantize_energy:
                    emac = (RoundSTE.apply(torch.exp(mod.act_noise.log_emac))).detach().cpu().numpy().flatten()
                else:
                    emac = (torch.exp(mod.act_noise.log_emac)).detach().cpu().numpy().flatten()

            stats["Avg Energy/MAC"].append(emac)
            stats["Neuron Dimension (D)"].append((np.ones_like(emac) * N).astype(int))
            stats["Layer"].append((np.ones_like(emac) * emacs_layer).astype(int))
            stats["Num Neurons (N')"].append((np.ones_like(emac) * num_neurons.item()).astype(int))

    if stats["Layer"]:
        return {stat_name: np.concatenate(stats[stat_name], axis=None) for stat_name in stats}
    return None 

def plot_emacs_per_dim(stats, model, run_name): 
    plot = sns.catplot(x="Neuron Dimension (D)", y="Avg Energy/MAC",
            data=pd.DataFrame(stats))
    plot.set(yscale="log")
    plot.fig.suptitle(f"{model}".capitalize() + f" Average Energy/MAC vs. Input Neurons")
    plot.set_xticklabels(rotation=45)
    y_minor = ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    plot.ax.yaxis.set_minor_locator(y_minor)
    plot.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    plt.savefig(f"{run_name}/{model}_emac_by_dim.pdf", bbox_inches="tight")
    plt.close()

    plot = sns.catplot(x="Num Neurons (N')", y="Avg Energy/MAC", 
            data=pd.DataFrame(stats))
    plot.set(yscale="log")
    plot.fig.suptitle(f"{model}".capitalize() + f" Average Energy/MAC vs. Output Neurons")
    plot.set_xticklabels(rotation=45)
    y_minor = ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    plot.ax.yaxis.set_minor_locator(y_minor)
    plot.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    plt.savefig(f"{run_name}/{model}_emac_by_num_neurons.pdf", bbox_inches="tight")
    plt.close()


    df = pd.DataFrame(stats)
    df["MACs"] = df["Num Neurons (N')"] * df["Neuron Dimension (D)"]
    plot = sns.catplot(x="MACs", y="Avg Energy/MAC", 
            data=df)
    plot.set(yscale="log")
    plot.fig.suptitle(f"{model}".capitalize() + f" Average Energy/MAC vs. MACs")
    plot.set_xticklabels(rotation=45)
    y_minor = ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    plot.ax.yaxis.set_minor_locator(y_minor)
    plot.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    plt.savefig(f"{run_name}/{model}_emac_by_macs.pdf", bbox_inches="tight")
    plt.close()

    plot = sns.relplot(x="Neuron Dimension (D)", y="Avg Energy/MAC", 
            data=pd.DataFrame(stats))
    plot.set(yscale="log")
    plot.set(xscale="log")
    plot.fig.suptitle(f"{model}".capitalize() + f" Average Energy/MAC vs. Input Neurons")
    plot.set_xticklabels(rotation=45)
    plt.savefig(f"{run_name}/{model}_emac_by_dim_relplot.pdf", bbox_inches="tight")
    plt.close()

    plot = sns.relplot(x="Num Neurons (N')", y="Avg Energy/MAC",  
            data=pd.DataFrame(stats))
    plot.set(yscale="log")
    plot.set(xscale="log")
    plot.fig.suptitle(f"{model}".capitalize() + f" Average Energy/MAC vs. Output Neurons")
    plot.set_xticklabels(rotation=45)
    plt.savefig(f"{run_name}/{model}_emac_by_num_neurons_relplot.pdf", bbox_inches="tight")
    plt.close()

    plot = sns.relplot(x="MACs", y="Avg Energy/MAC",  
            data=df)
    plot.set(yscale="log")
    plot.set(xscale="log")
    plot.fig.suptitle(f"{model}".capitalize() + f" Average Energy/MAC vs. MACs")
    plot.set_xticklabels(rotation=45)
    plt.savefig(f"{run_name}/{model}_emac_by_macs_relplot.pdf", bbox_inches="tight")
    plt.close()

def plot_overall_stat(stats, model, stat_name, run_name):
    print(f"Plotting {stat_name} histplot")
    plt.figure()
    plot = sns.histplot(stats["snr"], stat='probability')
    plot.set_title(f"{model}".capitalize() + f" {stat_name}")
    plot.set(xlabel=stat_name)
    plt.savefig(f"{run_name}/{model}_{stat_name}.pdf")
    plt.close()

    df = pd.DataFrame({"Neuron Dimension (D)": stats["dims"], stat_name: stats["snr"], 
        "Layer": stats["layers"], "Num Neurons": stats["num_neurons"]})

    print(f"Plotting {stat_name} scatterplots")
    plt.figure()
    plot = sns.catplot(x="Neuron Dimension (D)", y=stat_name, kind="violin",
            data=df)
    plot.fig.suptitle(f"{model}".capitalize() + f" {stat_name} by Dimension")
    plot.set_xticklabels(rotation=45)
    plt.savefig(f"{run_name}/{stat_name}_by_dim.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plot = sns.catplot(x="Layer", y=stat_name, kind="violin",
            data=df)
    plot.fig.suptitle(f"{model}".capitalize() + f" {stat_name} by Layer")
    plot.set_xticklabels(rotation=45)
    plt.savefig(f"{run_name}/{stat_name}_by_layer.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plot = sns.catplot(x="Num Neurons", y=stat_name, kind="violin",
            data=df)
    plot.fig.suptitle(f"{model}".capitalize() + f" {stat_name} by Num Neurons")
    plot.set_xticklabels(rotation=45)
    plt.savefig(f"{run_name}/{stat_name}_by_num_neurons.pdf", bbox_inches="tight")
    plt.close()

def plot_snr_before_after(model, before, after, run_name):
    data = pd.DataFrame({"SNR Before Training": before, "SNR After Training": after, "Change in SNR": after - before})

    plt.figure()
    plot = sns.displot(data=data, x="SNR Before Training", y="Change in SNR", kind='hist', cbar=True)
    plot.fig.suptitle(f"{model}".capitalize() + f" SNR Change by Training")
    plt.savefig(f"{run_name}/snr_change.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plot = sns.displot(data=data, x="SNR Before Training", y="SNR After Training", kind='hist', cbar=True)
    plot.fig.suptitle(f"{model}".capitalize() + f" SNR Before and After Training")
    plt.savefig(f"{run_name}/snr_before_and_after.pdf", bbox_inches="tight")
    plt.close()


def plot_noise_bits(model, noise_bits, run_name, setting):
    print(f"Plotting noise bits {setting}")
    plt.figure(figsize=(5, 3.5))
    plot = sns.lineplot(x=range(len(noise_bits)), y=noise_bits)
    plot.set_title(f"{model}".capitalize() + f" Noise Bits by Layer, {setting}")
    plot.set(xlabel="Layer", ylabel=f"Noise Bits")
    plt.yticks([3, 4, 5, 6, 7])
    plt.savefig(f"{run_name}/noise_bits_{setting.split()[0]}.pdf", bbox_inches="tight")
    plt.close()


def plot_emacs(model, run_name, model_name, noise_type):
    emac = 0
    noise = 0
    emacs = []
    denominator = 0
    for name, param in model.named_parameters():
        if "noise" in name:
            mod = model
            for subname in name.split('.')[:-1]:
                mod = getattr(mod, subname)

            e = torch.exp(param)
            if mod.quantize_energy:
                e = RoundSTE.apply(e)

            if noise_type == "shot":
                noise += 1. / torch.sqrt(e).mean() * mod.total_macs
            elif noise_type == "thermal":
                noise += 1. / torch.sqrt(e).mean() * mod.total_macs / 100.
            if noise_type == "weight":
                noise += 1. / torch.sqrt(e).mean() * mod.total_macs

            emac += e.mean() * mod.total_macs
            emacs.append(e.mean().item())
            denominator += mod.total_macs
    avg_emac = (emac / denominator).item()
    avg_noise = (noise / denominator).item()

    plt.figure(figsize=(5, 3.5))
    plot = sns.lineplot(x=range(len(emacs)), y=np.array(emacs) * 0.128)
    if model_name == "bert":
        title = f"BERT Energy/MAC by Matrix Multiplication"
        xlabel = "Matrix Multiplication"
    else:
        title = f"{model_name}".capitalize() + f" Energy/MAC by Layer" 
        xlabel = "Layer"

    sns.set_color_codes()
    plot.set_title(title)
    plot.set(yscale="log")
    plot.set(xlabel=xlabel, ylabel=f"Energy/MAC (aJ)")

    plot.yaxis.set_major_formatter(ticker.ScalarFormatter())
    y_minor = ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    plot.get_yaxis().set_minor_locator(y_minor)
    plot.grid(b=True, which='minor', color='lightgray', axis='y', linewidth=0.5)
    plot.yaxis.set_minor_locator(y_minor)
    plot.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.savefig(f"{run_name}/emac_by_layer.pdf", bbox_inches="tight")
    plt.close()

    return avg_emac, avg_noise


def get_bitwidths(module, round_bitwidth):
    stats = {"w_bits": [], "a_bits": [], "w_elems": []}

    for name, mod in module.named_children():
        if type(mod) not in MODULE_DICT.values():
            child_stats = get_bitwidths(mod, round_bitwidth)
            if child_stats is not None:
                for stat_name in stats:
                    stats[stat_name].extend(child_stats[stat_name])
        elif type(mod) not in [FloatFunctional, QInteractionLayer, QEinsum, QSoftmax, QEmbedding, QLayerNorm, QAdaptiveAvgPool2d, QAvgPool2d, QuantStub]:
            w_obs = mod.weight_quantizer.observer
            stats["w_bits"].append(torch.mean((w_obs.bitwidth_for_penalty)).item())
            stats["w_elems"].append(torch.prod(torch.tensor(mod.weight_quantizer.input_shape)))

            a_obs = mod.activation_quantizer.observer
            # TODO: skips float functionals which is fine usually but not great for ReLU quant
            if hasattr(a_obs, "bitwidth_for_penalty"):
                stats["a_bits"].append(torch.mean((a_obs.bitwidth_for_penalty)).item())

    return stats

def plot_bitwidths(model, run_name, mode, model_name, round_bitwidth):
    bitwidths = get_bitwidths(model, round_bitwidth)

    plt.figure()
    plot = sns.lineplot(x=range(len(bitwidths["w_bits"])), y=bitwidths["w_bits"])
    plot.set_title(f"{model_name}".capitalize() + f" Weight Bits by Layer {mode}")
    plot.set(xlabel="Layer")
    plt.savefig(f"{run_name}/wb_{mode}.pdf")
    plt.close()

    plt.figure()
    plot = sns.lineplot(x=range(len(bitwidths["a_bits"])), y=bitwidths["a_bits"])
    plot.set_title(f"{model_name}".capitalize() + f" Activation Bits by Layer {mode}")
    plot.set(xlabel="Layer")
    plt.savefig(f"{run_name}/ab_{mode}.pdf")
    plt.close()
