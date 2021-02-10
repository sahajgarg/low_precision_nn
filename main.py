import argparse
import shutil
import json

from itertools import product
from functools import partial
import tensorboardX

from utils import *
from data_loaders import *
from observers import *
from quantizers import *
from noise_models import *
from models.bert import Einsum

from evaluate import evaluate, evaluate_snr_cascading
from train import train, compute_bit_regularizer
from plot import *

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# Finalize observer calibration, and fix quantizer parameters
def finalize_calibration(model):
    for name, mod in model.named_children():
        recurse = True
        if hasattr(mod, 'activation_quantizer') and hasattr(mod.activation_quantizer, 'observer') and \
                hasattr(mod.activation_quantizer.observer, 'finalize_calibration'):
            mod.activation_quantizer.observer.finalize_calibration()
            recurse = False
        if hasattr(mod, 'weight_quantizer') and hasattr(mod.weight_quantizer, 'observer') and \
                hasattr(mod.weight_quantizer.observer, 'finalize_calibration'):
            mod.weight_quantizer.observer.finalize_calibration()
            recurse = False

        if recurse:
            finalize_calibration(mod)

# Evaluate the ratio of range for symmetric quantizers for weights versus asymmetric quantizers
def symmetric_weight_compression(model):
    sym_comp_tot = 0
    cnt = 0
    for name, mod in model.named_children():
        if hasattr(mod, 'weight_quantizer') and hasattr(mod.weight_quantizer, 'observer') and \
                hasattr(mod.weight_quantizer.observer, 'finalize_calibration'):
            sym_comp = mod.weight_quantizer.observer.symmetric_compression()
            print(f"Symmetric compression :\t\t{sym_comp.mean().item():0.2f},\t{torch.log2(sym_comp).mean().item():0.2f}")
            w_elems = torch.prod(torch.tensor(mod.weight_quantizer.input_shape))
            sym_comp_tot += torch.log2(sym_comp).mean() * w_elems
            cnt += w_elems
        else:
            child_sym_comp_tot, child_cnt = symmetric_weight_compression(mod)
            sym_comp_tot += child_sym_comp_tot
            cnt += child_cnt

    return sym_comp_tot, cnt

def get_quantizer(observer_name, bitwidth, symmetric, dynamic, batch_avg,
                  train_bitwidth, per_channel_bitwidth, train_qminmax,
                  train_dither, train_via_scale, round_bitwidth, discrete_bitwidth, stochastic_bitwidth, 
                  act_quant_before_relu, bias_correct, percentile):
    observer = None
    if observer_name == 'minmax':
        observer = partial(MinMaxObserver, bitwidth=bitwidth,
                symmetric=symmetric, dynamic=dynamic, batch_avg=batch_avg,
                train_bitwidth=train_bitwidth,
                per_channel_bitwidth=per_channel_bitwidth,
                train_qminmax=train_qminmax, train_via_scale=train_via_scale, 
                round_bitwidth=round_bitwidth, 
                discrete_bitwidth=discrete_bitwidth,
                stochastic_bitwidth=stochastic_bitwidth)
    elif observer_name == 'ema_minmax':
        observer = partial(EMAMinMaxObserver, bitwidth=bitwidth,
                symmetric=symmetric, dynamic=dynamic, batch_avg=batch_avg,
                train_bitwidth=train_bitwidth,
                per_channel_bitwidth=per_channel_bitwidth,
                train_qminmax=train_qminmax, train_via_scale=train_via_scale, 
                round_bitwidth=round_bitwidth, 
                discrete_bitwidth=discrete_bitwidth, 
                stochastic_bitwidth=stochastic_bitwidth)
    elif observer_name == 'channel_minmax':
        observer = partial(ChannelMinMaxObserver, bitwidth=bitwidth, 
                symmetric=symmetric,
                train_bitwidth=train_bitwidth,
                per_channel_bitwidth=per_channel_bitwidth,
                train_qminmax=train_qminmax, train_via_scale=train_via_scale, 
                round_bitwidth=round_bitwidth, 
                discrete_bitwidth=discrete_bitwidth,
                stochastic_bitwidth=stochastic_bitwidth)
    elif observer_name == 'channel_ema_minmax':
        observer = partial(EMAChannelMinMaxObserver, bitwidth=bitwidth, 
                symmetric=symmetric,
                train_bitwidth=train_bitwidth,
                per_channel_bitwidth=per_channel_bitwidth,
                train_qminmax=train_qminmax, train_via_scale=train_via_scale, 
                round_bitwidth=round_bitwidth, 
                discrete_bitwidth=discrete_bitwidth,
                stochastic_bitwidth=stochastic_bitwidth)
    elif observer_name == 'percentile_ema':
        observer = partial(EMAPercentileObserver, bitwidth=bitwidth,
                symmetric=symmetric, dynamic=dynamic, batch_avg=batch_avg,
                train_bitwidth=train_bitwidth,
                per_channel_bitwidth=per_channel_bitwidth,
                train_qminmax=train_qminmax, train_via_scale=train_via_scale, 
                round_bitwidth=round_bitwidth, 
                discrete_bitwidth=discrete_bitwidth,
                stochastic_bitwidth=stochastic_bitwidth, percentile=percentile)
    elif observer_name == 'percentile':
        observer = partial(PercentileObserver, bitwidth=bitwidth,
                symmetric=symmetric, dynamic=dynamic, batch_avg=batch_avg,
                train_bitwidth=train_bitwidth,
                per_channel_bitwidth=per_channel_bitwidth,
                train_qminmax=train_qminmax, train_via_scale=train_via_scale, 
                round_bitwidth=round_bitwidth, 
                discrete_bitwidth=discrete_bitwidth,
                stochastic_bitwidth=stochastic_bitwidth, percentile=percentile)
    elif observer_name == 'aciq':
        observer = partial(ACIQObserver, bitwidth, symmetric, dynamic, batch_avg)

    return partial(AffineQuantizer, observer,
            act_quant_before_relu=act_quant_before_relu,
            bias_correct=bias_correct, noise_only=(bitwidth == -1))


# Replace fused NN modules with their quantized counterparts 
def replace_modules(module, act_bitwidth, weight_bitwidth, e_mac, args,
        mapping=MODULE_DICT):
    for name, mod in module.named_children():
        if type(mod) not in mapping.keys():
            replace_modules(mod, act_bitwidth, weight_bitwidth, e_mac, args)
        else:
            if isinstance(mod, nn.Linear) and "channel" in args.weight_observer and e_mac < 0:
                # Noise experiments were done with per-channel linear as well.
                print("Overriding Linear observer with per tensor")
                weight_obs = "ema_minmax" 
                act_obs = "ema_minmax" if "channel" in args.act_observer else args.act_observer
            elif isinstance(mod, torch.quantization.QuantStub):
                # Use 8 bits for inputs
                # NOTE: this will not correctly train bitwidths for Shufflenet,
                # where QuantStub is also used after taking the mean for global pooling
                print("Using minmax for quant stub")
                act_obs = "minmax"
                weight_obs = args.weight_observer
                act_bitwidth = 8.
            else:
                weight_obs = args.weight_observer
                act_obs = args.act_observer

            weight_quantizer = get_quantizer(observer_name=weight_obs,
                    bitwidth=weight_bitwidth, symmetric=args.weight_symmetric,
                    dynamic=args.dynamic, batch_avg=args.batch_avg,
                    train_bitwidth=args.train_bitwidth,
                    per_channel_bitwidth=args.per_channel_bitwidth and "channel" in weight_obs,
                    train_qminmax=False,
                    train_dither=args.train_dither,
                    train_via_scale=args.train_via_scale,
                    round_bitwidth=args.round_bitwidth,
                    discrete_bitwidth=args.discrete_bitwidth,
                    stochastic_bitwidth=args.stochastic_bitwidth,
                    act_quant_before_relu=False,
                    bias_correct=args.bias_correct, percentile=args.percentile)

            if act_bitwidth == -1 and args.noise_type == "shot" and isinstance(mod, nnq.FloatFunctional):
                act_quantizer = nn.Identity
            else:
                act_quantizer = get_quantizer(observer_name=act_obs,
                        bitwidth=act_bitwidth, symmetric=args.act_symmetric,
                        dynamic=args.dynamic, batch_avg=args.batch_avg,
                        train_bitwidth=args.train_bitwidth,
                        per_channel_bitwidth=args.per_channel_bitwidth and "channel" in act_obs,
                        train_qminmax=args.train_qminmax,
                        train_dither=args.train_dither,
                        train_via_scale=args.train_via_scale,
                        round_bitwidth=args.round_bitwidth,
                        discrete_bitwidth=args.discrete_bitwidth,
                        stochastic_bitwidth=args.stochastic_bitwidth,
                        act_quant_before_relu=args.act_quant_before_relu,
                        bias_correct=False, percentile=args.percentile)

            if type(mod) in [nnq.FloatFunctional, nn.LayerNorm, nn.Softmax, nn.Embedding, 
                    nn.AdaptiveAvgPool2d, nn.AvgPool2d, torch.quantization.QuantStub] or e_mac <= 0:
                act_noise = nn.Identity()
                weight_noise = nn.Identity()
            else: 
                # Add noise to GEMM operations only
                assert isinstance(mod, nn.Linear) or type(mod) in [nni.ConvBn2d, nni.ConvBnReLU2d, nni.ConvReLU2d, Einsum]
                if isinstance(mod, nn.Linear):
                    act_shape = [1, mod.out_features] 
                    weight_shape = [mod.out_features, 1]
                elif type(mod) in [nni.ConvBn2d, nni.ConvBnReLU2d, nni.ConvReLU2d]:
                    act_shape = [1, mod[0].out_channels, 1, 1] 
                    weight_shape = [mod[0].out_channels, 1, 1, 1]
                if isinstance(mod, Einsum):
                    act_shape = [1]
                    weight_shape = None

                act_shape = act_shape if args.noise_per_channel else [1]
                act_noise = Gaussian(e_mac, args.noise_type, act_shape, args.record_snr, 
                        args.record_snr_cascading, args.quantize_energy) if args.noise_type != "weight" else nn.Identity()
                weight_shape = weight_shape if args.noise_per_channel else [1]
                weight_noise = Gaussian(e_mac, args.noise_type, weight_shape, args.record_snr, 
                        args.record_snr_cascading, args.quantize_energy) if args.noise_type == "weight" else nn.Identity()

            qconfig = {'weight': weight_quantizer, 'activation': act_quantizer, 
                    'act_noise': act_noise, 'weight_noise': weight_noise, 'train': False,
                    'quant_gemm_only': args.quant_gemm_only, 
                    'quant_relu_only': args.quant_relu_only}
            quantized_module = mapping[type(mod)].from_float(mod, qconfig=qconfig)
            setattr(module, name, quantized_module)

    return module


# Overriding with 8 bit quantizers for first and last layer. 
# This applies to the weights of the first and last layer as well as the 
# outputs of the first and last layer. 
def override_quantization(model, model_name, ignore_mode, train, args, act_bitwidth):
    weight_quantizer = get_quantizer(observer_name=args.weight_observer,
            bitwidth=8., symmetric=args.weight_symmetric, dynamic=args.dynamic,
            batch_avg=args.batch_avg, train_bitwidth=False,
            per_channel_bitwidth=False, train_qminmax=False,
            train_dither=args.train_dither,
            train_via_scale=args.train_via_scale, 
            round_bitwidth=args.round_bitwidth, 
            discrete_bitwidth=args.discrete_bitwidth,
            stochastic_bitwidth=args.stochastic_bitwidth,
            act_quant_before_relu=False,
            bias_correct=args.bias_correct, percentile=args.percentile)

    act_quantizer = get_quantizer(observer_name=args.act_observer, bitwidth=8.,
            symmetric=args.act_symmetric, dynamic=args.dynamic,
            batch_avg=args.batch_avg, train_bitwidth=False,
            per_channel_bitwidth=False, train_qminmax=False,
            train_dither=args.train_dither,
            train_via_scale=args.train_via_scale, 
            round_bitwidth=args.round_bitwidth,
            discrete_bitwidth=args.discrete_bitwidth,
            stochastic_bitwidth=args.stochastic_bitwidth,
            act_quant_before_relu=False,
            bias_correct=False, percentile=args.percentile)

    qconfig = {'weight': weight_quantizer, 'activation':
                act_quantizer, 'act_noise': nn.Identity(), 'weight_noise':
                nn.Identity(), 'train': False, 
                'quant_gemm_only': args.quant_gemm_only,
                'quant_relu_only': args.quant_relu_only}
    if "first" in ignore_mode:
        if model_name == "resnet50":
            conv1 = getattr(model, "conv1")
            conv1.set_qconfig(qconfig)
            if args.quant_gemm_only:
                # The activations from conv1 are quantized in the beginning of conv2, which is what must be ignored
                conv2 = getattr(getattr(model, "layer1")[0], "conv1")
                conv2.activation_quantizer = act_quantizer(shape=[1, conv2.in_channels, 1, 1], axis=1)
                downsample = getattr(getattr(model, "layer1")[0], "downsample")
                if downsample is not None:
                    print("Overriding input to downsample!")
                    downsample[0].activation_quantizer = act_quantizer(shape=[1, downsample[0].in_channels, 1, 1], axis=1)
        elif model_name == "resnet50_ptcv":
            conv1 = getattr(getattr(getattr(getattr(model, "features"), "init_block"), "conv"), "conv")
            conv1.set_qconfig(qconfig)
            if args.quant_gemm_only:
                raise NotImplementedError
        else: 
            raise NotImplementedError

    if "last" in ignore_mode:
        if args.quant_gemm_only:
            # The quant_gemm_only implementation uses act_quantizer for inputs to a layer. 
            # The outputs of the linear layer are automatically left unquantized.
            act_obs = "ema_minmax" if "channel" in args.act_observer else args.act_observer
            act_quantizer = get_quantizer(observer_name=act_obs,
                    bitwidth=act_bitwidth, symmetric=args.act_symmetric,
                    dynamic=args.dynamic, batch_avg=args.batch_avg,
                    train_bitwidth=args.train_bitwidth,
                    per_channel_bitwidth=args.per_channel_bitwidth and "channel" in act_obs,
                    train_qminmax=args.train_qminmax,
                    train_dither=args.train_dither,
                    train_via_scale=args.train_via_scale,
                    round_bitwidth=args.round_bitwidth,
                    discrete_bitwidth=args.discrete_bitwidth,
                    stochastic_bitwidth=args.stochastic_bitwidth,
                    act_quant_before_relu=args.act_quant_before_relu,
                    bias_correct=False, percentile=args.percentile)

            qconfig = {'weight': weight_quantizer, 'activation':
                        act_quantizer, 'act_noise': nn.Identity(), 'weight_noise':
                        nn.Identity(), 'train': False, 
                        'quant_gemm_only': args.quant_gemm_only,
                        'quant_relu_only': args.quant_relu_only}

        if model_name == 'resnet50' or model_name == "inceptionv3":
            model.fc.set_qconfig(qconfig)
        elif model_name == "resnet50_ptcv":
            model.output.set_qconfig(qconfig)
        else: 
            raise NotImplementedError  

    return model


# Evaluate the size of each feature map to determine the maximum size.
def get_act_map_sizes(module):
    stats = {"a_sizes": []}

    for name, mod in module.named_children():
        if type(mod) not in MODULE_DICT.values():
            child_stats = get_act_map_sizes(mod)
            if child_stats is not None:
                for stat_name in stats:
                    stats[stat_name].extend(child_stats[stat_name])
        elif hasattr(mod, "activation_quantizer") and not isinstance(mod.activation_quantizer, nn.Identity) and not isinstance(mod, QuantStub):
            a_elems = torch.prod(torch.tensor(mod.activation_quantizer.input_shape[1:]))
            a_bw = mod.activation_quantizer.observer.bitwidth
            stats["a_sizes"].append((a_bw * a_elems).item())

    return stats

# Set each layer bitwidth to be equivalent to the maximum activation map size
first_layer = True
def set_act_map_sizes(module, max_size, ignore):
    for name, mod in module.named_children():
        if type(mod) not in MODULE_DICT.values():
            set_act_map_sizes(mod, max_size, ignore)
        elif hasattr(mod, "activation_quantizer") and not isinstance(mod.activation_quantizer, nn.Identity) and not isinstance(mod, QuantStub):
            global first_layer
            if first_layer and "first" in ignore:
                print("Ignoring first layer")
                first_layer = False
            elif isinstance(mod, QLinear) and "last" in ignore:
                print("Ignoring last layer")
            else:
                a_elems = torch.prod(torch.tensor(mod.activation_quantizer.input_shape[1:]))
                a_bw = min(torch.floor(max_size / a_elems).item(), 32.)
                mod.activation_quantizer.observer.bitwidth = a_bw
                print(a_bw)

def quantize(model, train_loader, test_loader, act_bits, weight_bits, e_mac,
        args, lambd, target_emac, target_weight_bits, target_act_bits, tb_logger, plot_logdir, checkpoint=None):
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    if args.train:
        model.train()
    else:
        model.eval()

    # Replace nn modules with QAT modules
    model = replace_modules(model, act_bits, weight_bits, e_mac, args)
    model = override_quantization(model, args.model, args.ignore, args.train, args, act_bits)
    set_layers(model)
    model = model.to(args.device)
    
    # Perform calibration when needed, possibly to initialize the size of certain parameters
    if len(args.save_path) > 0:
        print("Calibrating for one batch to initialize certain parameters.")
        acc = evaluate(model, train_loader, 1, args.device, args)
    elif not args.dynamic and not (act_bits == -1 and weight_bits == -1 and args.noise_type == "shot"):
        print("Calibrating Model.")
        stop_noise(model)
        acc = evaluate(model, train_loader, args.calibration_batches, args.device, args)
        start_noise(model)
    else:
        print("Not calibrating.")
    finalize_calibration(model)

    # Evaluate the range ratio when using symmetric and asymmetric weights
    if args.eval_symmetric_weight_compression:
        sym_comp, cnt = symmetric_weight_compression(model)
        sym_comp /= cnt
        print(f"Average weight bit saving from asymmetric: {sym_comp}")

    # Set the bit precision of each layer to have each layer feature map size equal to the 
    # maximum feature map size.
    if args.max_act_analytical:
        global first_layer
        first_layer = True
        set_act_map_sizes(model, max(get_act_map_sizes(model)["a_sizes"]) * target_act_bits / act_bits, args.ignore)

    if args.train or args.train_noise or args.train_bitwidth or args.train_qminmax:
        # Either evaluate checkpointed model, or train the energy allocations if 
        # the model has not yet been checkpointed.
        if len(args.save_path) > 0 and checkpoint is None:
            model.load_state_dict(torch.load(f"{args.save_path}/model.pth"))
            if args.train_bitwidth and args.train_dither:
                relax_quantization(model)
        elif checkpoint == 'loss':
            print("Evaluating loss checkpoint")
            model.load_state_dict(torch.load(f"{plot_logdir}/model_loss.pth"))
            if args.train_bitwidth and args.train_dither:
                relax_quantization(model)
        elif checkpoint == 'accuracy':
            print("Evaluating accuracy checkpoint")
            try:
                if len(args.save_path) > 0:
                    model.load_state_dict(torch.load(f"{args.save_path}/model_acc.pth"))
                else:
                    model.load_state_dict(torch.load(f"{plot_logdir}/model_acc.pth"))
            except:
                print("No checkpoint saved, evaluating final model")
                model.load_state_dict(torch.load(f"{plot_logdir}/model.pth"))

            if args.train_bitwidth and args.train_dither:
                relax_quantization(model)
        else:
            # Set trainable parameters depending on flags.
            print("Beginning Training.")
            if args.train_noise:
                print("Training noise")
                for name, param in model.named_parameters():
                    if "noise" not in name:
                        param.requires_grad = False
            if args.train_bitwidth:
                print("Training bitwidth!")
                for name, param in model.named_parameters():
                    if "quantizer" not in name:
                        param.requires_grad = False
                    if "quantizer" in name and "embedding" in name:
                        print("Embeddings fixed at 8 bits!")
                        param.requires_grad = False
                    if "activation" in name and "bitwidth" in name and args.weight_bits_only:
                        param.requires_grad = False
                    if "weight" in name and "bitwidth" in name and args.act_bits_only:
                        param.requires_grad = False

                if args.train_dither:
                    relax_quantization(model)
            elif args.train_qminmax:
                print("Training qminmax with fixed bitwidth!")
                for name, param in model.named_parameters():
                    # only train min_val and max_val
                    if "_val" not in name:
                        param.requires_grad = False

            model = train(model, train_loader, test_loader, args, lambd, tb_logger, target_emac,
                          target_weight_bits, target_act_bits, plot_logdir)
            torch.save(model.state_dict(), f"{plot_logdir}/model.pth")

    return model


def get_logging_stats(model, args, e_mac, plot_logdir, tb_logger, step):
    mean_snr, avg_emac, avg_noise, w_bits, a_bits = None, None, None, None, None

    if e_mac > 0 and args.noise_type == "shot":
        a_bits = torch.tensor(-1)
        w_bits = torch.tensor(-1)
    else:
        w_bits, w_sum, a_bits, a_sum = compute_bit_regularizer(model, args.round_bitwidth, args.max_act or args.max_act_analytical)
        w_bits /= w_sum 
        a_bits /= a_sum 
        plot_bitwidths(model, plot_logdir, "rounded", args.model, args.round_bitwidth)

    if args.record_snr or args.record_snr_cascading:
        snr_stats = get_stats(model, args, args.override_with_noise_bits)
        plot_overall_stat(snr_stats, args.model, "SNR after training", plot_logdir)
        if "noise_bits" in snr_stats:
            setting = "Dynamic Energy/MAC" if args.train_noise else "Uniform Energy/MAC"
            plot_noise_bits(args.model, snr_stats["noise_bits"], plot_logdir, setting)
        emac_stats = get_emac_stats(model, args)
        plot_emacs_per_dim(emac_stats, args.model, plot_logdir)
        if args.record_snr_cascading:
            print(snr_stats["snr"])
            predicted_cascade = np.log10(1. / (np.cumsum(1. / np.emac(10., snr_stats["snr"] / 10.)))) * 10.
            print(snr_stats["snr_cascading"])
            print(predicted_cascade)
        mean_snr = np.mean(snr_stats["snr"])

    if e_mac > 0:
        avg_emac, avg_noise = plot_emacs(model, plot_logdir, args.model, args.noise_type)

    if args.train_noise:
        tb_logger.add_scalar('eval/avg_energy', avg_emac, global_step=step)
        tb_logger.add_scalar('eval/avg_noise', avg_noise, global_step=step)
    if args.train_bitwidth:
        tb_logger.add_scalar('eval/bits_weight', w_bits, global_step=step)
        tb_logger.add_scalar('eval/bits_act', a_bits, global_step=step)

    return mean_snr, avg_emac, avg_noise, w_bits, a_bits


# Generate plotting subdirectory for sweeps
def get_plot_logdir(act_bits, weight_bits, e_mac, args, lambd, target_emac,
        target_weight_bits, target_act_bits, train_subset):
    plot_subdir = ""
    if len(args.weight_bits) > 1:
        plot_subdir += f"w{weight_bits}"
    if len(args.act_bits) > 1:
        plot_subdir += f"a{act_bits}"
    if len(args.e_mac) > 1:
        plot_subdir += f"n{e_mac}"
    if len(args.lambd) > 1:
        plot_subdir += f"L{lambd}"
    if len(args.target_emac) > 1:
        plot_subdir += f"tp{target_emac}"
    if len(args.target_weight_bits) > 1:
        plot_subdir += f"tw{target_weight_bits}"
    if len(args.target_act_bits) > 1:
        plot_subdir += f"ta{target_act_bits}"
    if len(args.train_subset) > 1:
        plot_subdir += f"ts{train_subset}"
    if len(plot_subdir) > 0:
        plot_logdir = f"{args.run_name}/{plot_subdir}"
        print(f"Logging to {plot_logdir}")
    else:
        plot_logdir = args.run_name

    return plot_logdir

# Evaluate accuracy when bitwidths are allowed to be fractional -- not typically used.
def eval_dither(args, model, test_loader, tb_logger, step):
    if args.train_bitwidth and not args.round_bitwidth and not args.discrete_bitwidth and not args.stochastic_bitwidth:
        dither_acc = evaluate(model, test_loader, args.eval_batches, args.device, args)
        w_bits_dither, w_sum, a_bits_dither, a_sum = compute_bit_regularizer(model, 
                args.round_bitwidth, args.max_act or args.max_act_analytical)
        w_bits_dither = (w_bits_dither / w_sum).item()
        a_bits_dither = (a_bits_dither / a_sum).item()
        plot_bitwidths(model, plot_logdir, "dithered", args.model, args.round_bitwidth)
        finalize_bitwidth(model)

        tb_logger.add_scalar('eval_dither/acc', dither_acc, global_step=step)
        tb_logger.add_scalar('eval_dither/bits_weight', w_bits_dither, global_step=step)
        tb_logger.add_scalar('eval_dither/bits_act', a_bits_dither, global_step=step)
    else:
        w_bits_dither, a_bits_dither, dither_acc = None, None, None
    return w_bits_dither, a_bits_dither, dither_acc

# When evaluating subject to noise, change noise to noise equivalent bit precision
def override_with_noise_bits(args, model, test_loader):
    if args.override_with_noise_bits:
        stop_noise(model)
        stop_recorders(model)
        noise_bits_acc = evaluate(model, test_loader, args.eval_batches, args.device, args)
        w_bits_noise, w_sum, a_bits_noise, a_sum = compute_bit_regularizer(model, 
                args.round_bitwidth, args.max_act or args.max_act_analytical, fixed_bitwidth=True, ignore_elemwise=True)
        w_bits_noise = (w_bits_noise / w_sum).item()
        a_bits_noise = (a_bits_noise / a_sum).item()
    else: 
        noise_bits_acc, w_bits_noise, a_bits_noise = None, None, None
    return noise_bits_acc, w_bits_noise, a_bits_noise

# Evaluate floating point baseline with no noise.
def run_fp_baseline(args, num_labels, train_loader, test_loader, lambd):
    model = load_model(args.model, args.device, args.dataset, num_labels)
    if args.train:
        train(model, train_loader, test_loader, nn.Identity(), args, lambd)
    acc = evaluate(model, test_loader, args.eval_batches, args.device, args)
    if args.train:
        model = train_imagenet(model, train_loader, test_loader, args, None, None, None, None)
        acc = evaluate(model, test_loader, args.eval_batches, args.device, args)
    return acc, None, None, None, torch.tensor(-1), torch.tensor(-1), None, None, None, None, None, None

def run_quantization(act_bits, weight_bits, e_mac, args, train_loader,
        test_loader, lambd, target_emac, target_weight_bits, target_act_bits, train_subset, checkpoint=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.model == 'bert':
        label_list = train_loader.dataset.features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = None

    # Evaluate floating point baseline with no noise.
    if act_bits == -1 and weight_bits == -1 and e_mac == -1:
        return run_fp_baseline(args, num_labels, train_loader, test_loader, lambd)

    # Perform quantization
    model = load_model_quantized(args.model, args.device, args.dataset, num_labels)
    plot_logdir = get_plot_logdir(act_bits, weight_bits, e_mac, args, lambd,
            target_emac, target_weight_bits, target_act_bits, train_subset)
    tb_logger = tensorboardX.SummaryWriter(log_dir=plot_logdir)
    model = quantize(model, train_loader, test_loader, act_bits, weight_bits,
            e_mac, args, lambd, target_emac, target_weight_bits,
            target_act_bits, tb_logger, plot_logdir, checkpoint)

    print("Evaluating.")
    step = args.epochs * len(train_loader) if args.train or args.train_noise or args.train_bitwidth else 0
    if args.record_snr:
        start_recorders(model)
        reset_stats(model)
    w_bits_dither, a_bits_dither, dither_acc = eval_dither(args, model, test_loader, tb_logger, step)
    if args.record_snr_cascading:
        acc = evaluate_snr_cascading(model, test_loader, args.eval_batches, args.device)
    else:
        acc = evaluate(model, test_loader, args.eval_batches, args.device, args)
    tb_logger.add_scalar('eval/acc', acc, global_step=step)

    # Save model outputs for pseudolabels, if needed
    if args.save_model_outputs is not None:
        save_outputs(model, train_loader, args.train_batches, args.device, args.save_model_outputs)

    mean_snr, avg_emac, avg_noise, w_bits, a_bits = get_logging_stats(model, args, e_mac, plot_logdir, tb_logger, step)
    noise_bits_acc, w_bits_noise, a_bits_noise = override_with_noise_bits(args, model, test_loader)

    return acc, mean_snr, avg_emac, avg_noise, w_bits, a_bits, w_bits_dither,\
            a_bits_dither, dither_acc, noise_bits_acc, w_bits_noise, a_bits_noise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet50', 'inceptionv3', 'mobilenet', 
        'dlrm', 'bert', 'shufflenetv2', 'googlenet', 'resnet50_ptcv'],
                        default='resnet50')
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'cifar10', 'criteo', 'mnli'],
                        default='imagenet')
    parser.add_argument('--data_path', type=str, default='/mnt/efs/')
    parser.add_argument('--run_name', type=str, default='',
                        help='Experiment Name for saving results')
    parser.add_argument('--save_path', type=str, default='',
                        help='Load model and args from save_path. Some fields will be retained from old args.')
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--act_bits', type=float, default=[8], nargs='*',
                        help='The number of bits to quantize activations to. Pass a list to sweep'
                             'over multiple settings. Can pass non-integer to use round(2**act_bits) bins.')
    parser.add_argument('--dynamic', action='store_true', help='Dynamically compute quantization parameters'
                                                               'instead of using calibration dataset.')
    parser.add_argument('--batch_avg', action='store_true',
                        help='Compute quantization parameters per-instance and average over the batch.')
    parser.add_argument('--percentile', type=float, default=99.99, help="percentile for clipping when using percentile_ema")
    parser.add_argument('--weight_bits', type=float, default=[8], nargs='*',
                        help='The number of bits to quantize weights to. Pass a list to sweep'
                             'over multiple settings. Can pass non-integer to use round(2**act_bits) bins.')
    parser.add_argument('--ignore', type=str, choices=['first', 'last', 'first+last', 'None'],
                        default='None', help="Specify layers to leave at full precision.")
    parser.add_argument('--act_observer', type=str, choices=['minmax', 'ema_minmax', 'histogram', 'aciq', 
                        'channel_minmax', 'percentile_ema', 'percentile', 'channel_ema_minmax'],
                        default='ema_minmax')
    parser.add_argument('--act_symmetric', action='store_true')
    parser.add_argument('--quant_gemm_only', action='store_true', help='Quantize only GEMM inputs, i.e. do not quantize skip and residual')
    parser.add_argument('--quant_relu_only', action='store_true', help='Quantize only ReLUs, i.e. do not quantize residuals')
    parser.add_argument('--max_act', action='store_true', help='Penalize maximum activation map size instead of average')
    parser.add_argument('--max_act_analytical', action='store_true', help='Set each intermediate feature map size equal to the maximum size')
    parser.add_argument('--act_quant_before_relu', action='store_true')
    parser.add_argument('--bias_correct', action='store_true')
    parser.add_argument('--weight_observer', type=str, default='channel_ema_minmax',
                        choices=['minmax', 'ema_minmax', 'channel_minmax', 'channel_ema_minmax', 'histogram'])
    parser.add_argument('--weight_symmetric', action='store_true')
    parser.add_argument('--calibration_batches', type=int, default=100)
    parser.add_argument('--eval_batches', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--evaluate_steps', type=int, default=-1, 
                        help='Number of steps to evaluate model at when training')
    parser.add_argument('--eval_symmetric_weight_compression', action='store_true', help='Eval symmetric weight comp')

    parser.add_argument('--record_snr', action='store_true', help='Record SNR and noise bits.')
    parser.add_argument('--record_snr_cascading', action='store_true', help='Record SNR cascading noise.')
    parser.add_argument('--quantize_energy', action='store_true', help='Only allow for quantized energy levels.')
    parser.add_argument('--override_with_noise_bits', action='store_true', help='Compute noise bits, then override bitwidth for second eval.')
    parser.add_argument('--e_mac', type=float, default=[-1], nargs='*',
                        help='Energy per MAC when evaluating with noise.')
    parser.add_argument('--noise_type', type=str, default="shot", choices=['shot', 'thermal', 'weight'],
                        help='Noise type, shot or thermal.')
    
    parser.add_argument('--target_weight_bits', type=float, default=[4], nargs='*', 
                        help='The number of target bits to quantize weights to')
    parser.add_argument('--target_act_bits', type=float, default=[4], nargs='*', 
                        help='The number of target bits to quantize activations to')
    parser.add_argument('--target_emac', type=float, default=[4.], nargs='*', 
                        help='The target number of photons per MAC')
    parser.add_argument('--constrained_loss', action='store_true', help='Constrained max margin loss')

    parser.add_argument('--train', action='store_true', help='Enable quantization aware training')
    parser.add_argument('--train_batches', type=int, default=10000, help='Number of batches to train on')
    parser.add_argument('--train_noise', action='store_true', help='Train noise standard deviation')
    parser.add_argument('--train_bitwidth', action='store_true', help='Train bitwidths')
    parser.add_argument('--weight_bits_only', action='store_true', help='Train weight bitwidths only')
    parser.add_argument('--act_bits_only', action='store_true', help='Train act bitwidths only')
    parser.add_argument('--train_via_scale', action='store_true', help='Train bitwidths via scale parameter')
    parser.add_argument('--train_dither', action='store_true', help='Train by dithering, not directly')
    parser.add_argument('--noise_per_channel', action='store_true', help='Train noise with per channel param')
    parser.add_argument('--per_channel_bitwidth', action='store_true', help='Learn per channel bitwidths')
    parser.add_argument('--train_qminmax', action='store_true', help='Train max and min for quantization')

    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer to use.')
    parser.add_argument('--lambd', type=float, default=[0.1], nargs='*',
                        help='Lambda for bitwidth or noise regularizer')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--kure', action='store_true', help='Kurtosis regularization')

    parser.add_argument('--train_subset', type=int, default=[128118], nargs='*')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--save_model_outputs', type=str, default=None)

    parser.add_argument('--round_bitwidth', action='store_true', help='Round bitwidth during training')
    parser.add_argument('--discrete_bitwidth', action='store_true', help='Round bitwidth to discrete set during training')
    parser.add_argument('--stochastic_bitwidth', action='store_true', help='Stochastically round bitwidth during training')

    parser.add_argument('--binary_search_acc', type=float,
                        help='Binary search emac for target accuracy')
    parser.add_argument('--search_min_emac', type=float, 
                        help='Min emac for binary search. Should yield acc less than binary_search_acc')
    parser.add_argument('--search_max_emac', type=float, 
                        help='Max emac for bianry search. Should yield acc greater than binary_search_acc')

    args = parser.parse_args()

    assert len(args.run_name) > 0, "Must specify run name for logging results" 
    assert args.run_name != args.save_path, "Must log to different directory than model loading"
    args.run_name = f"logs/{args.run_name}"
    if os.path.exists(f"{args.run_name}"):
        print(f"Overwriting results at {args.run_name}")
        shutil.rmtree(args.run_name)

    os.makedirs(args.run_name)
    sns.set_style("whitegrid")

    print(f"Logging to {args.run_name}")

    if len(args.save_path) > 0:
        args.save_path = f"logs/{args.save_path}"
        print(f"Reading results from {args.save_path}")
        with open(f"{args.save_path}/args.txt", 'r') as f:
            saved_args_dict = eval(f.read())
            saved_args = argparse.Namespace(**saved_args_dict)
        saved_args.save_path = args.save_path
        saved_args.run_name = args.run_name
        saved_args.data_path = args.data_path
        saved_args.eval_batches = args.eval_batches
        saved_args.val_batch_size = args.val_batch_size
        saved_args.val_data_path = args.val_data_path
        args = saved_args

    if args.model == "dlrm":
        args.dataset = "criteo"
    elif args.model == "bert":
        args.dataset = "mnli"

    # ecaq folds per-channel activation quantizer parameters into weight quantizer 
    # parameters in the next layer, analogous to batch norm folding
    # Presented in: http://proceedings.mlr.press/v119/wang20c/wang20c.pdf 
    args.ecaq = "channel" in args.act_observer
    if args.ecaq:
        assert args.quant_gemm_only, "ecaq only implemented for quant_gemm_only"

    with open(f"{args.run_name}/args.txt", 'w') as f:
        f.write(str(vars(args)))

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using " + str(args.device))

    return args

# Sweep over many different targets and run quantization for each of those targets
def sweep(args, data):
    i = 0
    prev_train_subset = None
    for act_bitwidth, weight_bitwidth, e_mac, lambd, target_emac, target_weight_bits, target_act_bits, train_subset in \
            product(list(args.act_bits), list(args.weight_bits),
                    list(args.e_mac), list(args.lambd),
                    list(args.target_emac), list(args.target_weight_bits),
                    list(args.target_act_bits), list(args.train_subset)):
        
        if args.act_observer == 'aciq':
            assert (act_bitwidth == round(act_bitwidth) and weight_bitwidth == round(weight_bitwidth))
            act_bitwidth = round(act_bitwidth)
            weight_bitwidth = round(weight_bitwidth)
        if args.noise_type == "thermal":
            assert (weight_bitwidth > 0 and act_bitwidth > 0), "Thermal noise only when quantizing"
        if args.noise_type == "weight":
            assert weight_bitwidth > 0, "Weight noise only when quantizing."
     
        if args.model == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-cased',
                use_fast=True,
            )
        else:
            tokenizer = None

        # Don't re-load data unless train subset has changed or at the beginning
        if prev_train_subset is None or train_subset != prev_train_subset:
            print("Preparing data loaders.")
            train_loader, test_loader, num_epochs = prepare_data_loaders(args.dataset, args.data_path, args.train_batch_size,
                                                                        args.val_batch_size, args.train_data_path, args.val_data_path, 
                                                                        args, train_subset, tokenizer)
            if num_epochs != -1:
                args.epochs = num_epochs
            print("Loaded data.")
            prev_train_subset = train_subset

        if args.checkpoint:
            checkpointed_models = [None, "accuracy"]
        else:
            checkpointed_models = [None]

        # Checkpointed model implemented as a hack - reloads a saved model.
        for chk_model in checkpointed_models:
            acc, mean_snr, avg_emac, avg_noise, w_bits, a_bits, w_bits_dither, a_bits_dither, dither_acc, noise_bits_acc, w_bits_noise, a_bits_noise = \
                run_quantization(act_bitwidth, weight_bitwidth, e_mac, args, train_loader, test_loader, 
                        lambd, target_emac, target_weight_bits, target_act_bits,train_subset, chk_model)

            row = []
            row.extend([w_bits.item(), a_bits.item()])
            if args.train_bitwidth or args.train_noise:
                row.extend([lambd])
            if args.record_snr:
                row.append(mean_snr)
            if args.e_mac[0] != -1:
                row.extend([avg_noise, avg_emac])
            if args.train_bitwidth and not args.round_bitwidth and not args.discrete_bitwidth and not args.stochastic_bitwidth:
                row.extend([w_bits_dither, a_bits_dither, dither_acc])
            if args.override_with_noise_bits:
                row.extend([w_bits_noise, a_bits_noise, noise_bits_acc])
            row.append(acc)

            data.loc[i] = row
            i += 1
            print(row)
    
    return data

# Perform binary search for a target energy/MAC for noise evaluation. 
def binary_search(args, data):
    print("Binary Searching!")
     
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-cased',
            use_fast=True,
        )
    else:
        tokenizer = None

    print("Preparing data loaders.")
    train_loader, test_loader, num_epochs = prepare_data_loaders(args.dataset, args.data_path, args.train_batch_size,
                                                 args.val_batch_size, args.train_data_path, args.val_data_path, args, args.train_subset[0], tokenizer)
    if num_epochs != -1:
        args.epochs = num_epochs
    print("Loaded data.")

    min_emac = args.search_min_emac
    max_emac = args.search_max_emac
    weight_bitwidth = args.weight_bits[0]
    act_bitwidth = args.act_bits[0]
    e_mac = args.e_mac[0]
    lambd = args.lambd[0]
    if args.noise_type == "thermal":
        assert (weight_bitwidth > 0 and act_bitwidth > 0), "Thermal noise only when quantizing"
    if args.noise_type == "weight":
        assert weight_bitwidth > 0, "Weight noise only when quantizing."

    i = 0
    while True:
        if i > 15:
            return data
        mid = np.exp((np.log(min_emac) + np.log(max_emac)) / 2)
        if not args.train_noise: 
            e_mac = mid

        print(f"Binary search step {i} with target emac {mid}")
        acc, mean_snr, avg_emac, avg_noise, w_bits, a_bits, w_bits_dither, a_bits_dither, dither_acc, noise_bits_acc, w_bits_noise, a_bits_noise = \
                run_quantization(act_bitwidth, weight_bitwidth, e_mac, args, train_loader, test_loader, lambd, mid, None, None, None, [None])

        row = []
        row.extend([w_bits.item(), a_bits.item()])
        if args.train_bitwidth or args.train_noise:
            row.extend([lambd])

        if args.record_snr or args.record_snr_cascading:
            row.append(mean_snr)
        if args.e_mac[0] != -1:
            row.extend([avg_noise, avg_emac])
        if args.train_bitwidth and not args.round_bitwidth and not args.discrete_bitwidth and not args.stochastic_bitwidth:
            row.extend([w_bits_dither, a_bits_dither, dither_acc])
        if args.override_with_noise_bits:
            row.extend([w_bits_noise, a_bits_noise, noise_bits_acc])
        row.append(acc)

        data.loc[i] = row
        print(f"Binary search step {i} obtained emac {avg_emac} at accuracy {acc}")
        print(row)
        i += 1

        if acc > args.binary_search_acc and (acc - args.binary_search_acc) < 0.1:
            break
        elif acc > args.binary_search_acc:
            max_emac = avg_emac 
        else:
            min_emac = avg_emac
    
    return data

def main():
    args = parse_args()

    columns = ['Weight Bits', 'Activation Bits']
    if args.train_bitwidth:
        columns.append('Bitwidth Lambda')
    if args.train_noise:
        columns.append('Noise Lambda')
    if args.record_snr or args.record_snr_cascading:
        columns.append('Mean SNR')
    if args.e_mac[0] != -1:
        columns.extend(['Avg Noise', 'Avg Power'])
    if args.train_bitwidth and not args.round_bitwidth and not args.discrete_bitwidth and not args.stochastic_bitwidth:
        columns.extend(['Wbits Dither', 'Abits Dither', 'Dither Acc'])
    if args.override_with_noise_bits:
        columns.extend(['Noise Weight Bits', 'Noise Act Bits', 'Noise Bits Acc'])
    if args.model == "dlrm":
        columns.append('auroc')
    else:
        columns.append('Accuracy')
    data = pd.DataFrame(columns=columns)

    if args.binary_search_acc is None:
        data = sweep(args, data)
    else:
        data = binary_search(args, data)

    print(data)
    data.to_csv(f"{args.run_name}/results.csv")
    print(f"Results saved at {args.run_name}/results.csv")


if __name__ == '__main__':
    main()
