import sys
from math import exp
from tqdm import tqdm
from itertools import chain
from sklearn.metrics import roc_auc_score
from datasets import load_metric
from transformers import Trainer, default_data_collator
import tensorboardX

from utils import *
from quantizers import RoundSTE
from evaluate import evaluate, evaluate_imagenet, evaluate_dlrm


def kurtosis_loss(model, loss, tot):
    for name, mod in model.named_children():
        if hasattr(mod, 'weight'):
            mean = torch.mean(mod.weight)
            sd = std(mod.weight)
            diffs = (mod.weight - mean) / sd
            kurtosis = torch.mean(diffs.pow(4))
            loss = loss + (kurtosis - 1.8).pow(2)
            tot += 1
        else:
            loss, tot = kurtosis_loss(mod, loss, tot)
    return loss, tot

def train(model, train_loader, test_loader, args, lambd, tb_logger,
        target_emac=None, target_weight_bits=8,
        target_act_bits=8, plot_logdir=None):
    if args.dataset == "imagenet":
        return train_imagenet(model, train_loader, test_loader, args, lambd,
                tb_logger, target_emac, target_weight_bits, target_act_bits,
                plot_logdir)
    elif isinstance(model, DLRM_Net):
        return train_dlrm(model, train_loader, test_loader, args, lambd, 
                target_weight_bits, target_act_bits)
    else:
        return train_bert(model, train_loader, test_loader, args, lambd, 
                target_emac, target_weight_bits, target_act_bits)


def train_dlrm(model, train_loader, test_loader, args, lambd, target_weight_bits=8, target_act_bits=8):
    criterion = torch.nn.BCELoss(reduction="mean")
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        loss_ema = None
        auroc_ema = None
        num_batches = len(train_loader) if args.train_batches == -1 else min(len(train_loader), args.train_batches)
        pbar = tqdm(total=num_batches, file=sys.stdout, leave=False)

        for it, (X, lS_o, lS_i, T) in enumerate(train_loader):
            if it > args.train_batches:
                break
            optimizer.zero_grad()
            lS_i = [S_i.to(args.device) for S_i in lS_i] if isinstance(lS_i, list) \
                else lS_i.to(args.device)
            lS_o = [S_o.to(args.device) for S_o in lS_o] if isinstance(lS_o, list) \
                else lS_o.to(args.device)

            Z = model(X.to(args.device), lS_o, lS_i)
            loss = criterion(Z, T.to(args.device))

            if args.kure:
                kurt_loss, num_weights = kurtosis_loss(model, 0., 0)
                loss = loss + kurt_loss / num_weights
            if args.train_noise:
                emac = compute_noise_regularizer(model)
                loss = loss + lambd * torch.log(emac)
            if args.train_bitwidth:
                w_bits, w_sum, a_bits, a_sum = compute_bit_regularizer(model, args.round_bitwidth, args.max_act)
                w_bits /= w_sum 
                a_bits /= a_sum
                if args.constrained_loss:
                    bit_loss = lambd * (torch.clamp(w_bits - target_weight_bits, min=0.)
                                             + torch.clamp(a_bits - target_act_bits, min=0.))
                else:
                    bit_loss = lambd * (w_bits + a_bits)
                loss += bit_loss 

            loss.backward()
            optimizer.step()

            auroc = roc_auc_score(T.detach().cpu().numpy(), Z.detach().cpu().numpy())
            loss_ema = loss if loss_ema is None else 0.1 * loss + 0.9 * loss_ema
            auroc_ema = auroc if auroc_ema is None else 0.1 * auroc + 0.9 * auroc_ema

            if args.train_noise:
                pbar.set_description("Epoch {:d}, loss {:.4f}, train auroc {:.4f}, emac {:.4f}"\
                        .format(epoch, loss_ema, auroc_ema, emac))
            elif args.train_bitwidth:
                pbar.set_description("Epoch {:d}, loss {:.4f}, train auroc {:.4f}, w_bits {:.4f}, a_bits {:.4f}"\
                        .format(epoch, loss_ema, auroc_ema, w_bits, a_bits))
            else:
                pbar.set_description("Epoch {:d}, loss {:.4f}, train auroc {:.4f}"\
                          .format(epoch, loss_ema, auroc_ema))
            pbar.update(1)

        pbar.close()
        val_acc = evaluate_dlrm(model, test_loader, args.eval_batches, args.device)

    return model


def compute_noise_regularizer(model):
    emac = 0
    denominator = 0
    for name, param in model.named_parameters():
        if "noise" in name:
            mod = model
            for subname in name.split('.')[:-1]:
                mod = getattr(mod, subname)

            if mod.quantize_energy:
                emac += torch.exp(param).mean() * mod.total_macs
            else:
                emac += RoundSTE.apply(torch.exp(param)).mean() * mod.total_macs
            denominator += mod.total_macs

    avg_emac = emac / denominator
    return avg_emac

def compute_bit_regularizer(module, round_bitwidth, max_act, fixed_bitwidth=False, tb_logger=None, step=0, ignore_elemwise=False):
    w_bits = 0
    w_sum = 0
    a_bits = 0
    a_sum = 0
    for name, mod in module.named_children():
        if type(mod) not in MODULE_DICT.values():
            child_w_bits, child_w_sum, child_a_bits, child_a_sum = \
                    compute_bit_regularizer(mod, round_bitwidth, max_act, fixed_bitwidth, tb_logger, step, ignore_elemwise)
            w_bits += child_w_bits 
            w_sum += child_w_sum 
            if max_act:
                if child_a_bits > a_bits:
                    a_bits = child_a_bits
                    a_sum = child_a_sum
            else:
                a_bits += child_a_bits 
                a_sum += child_a_sum
        else:
            if type(mod) not in [QEmbedding, QuantStub]:
                if not isinstance(mod.activation_quantizer, nn.Identity) and mod.activation_quantizer.input_shape: 
                    if not ignore_elemwise or type(mod) not in [FloatFunctional, 
                            QInteractionLayer, QEinsum, QSoftmax, QEmbedding, QLayerNorm, QAdaptiveAvgPool2d, QAvgPool2d]:
                        a_obs = mod.activation_quantizer.observer
                        a_elems = torch.prod(torch.tensor(mod.activation_quantizer.input_shape[1:]))
                        a_bw = a_obs.bitwidth_for_penalty
                        if max_act:
                            if torch.mean(a_bw) * a_elems > a_bits:
                                a_bits = torch.mean(a_bw) * a_elems
                                a_sum = a_elems
                        else:
                            a_bits += torch.mean(a_bw) * a_elems
                            a_sum += a_elems
                        if tb_logger: 
                            tb_logger.add_scalar(f"a_bits/{mod.layer_num}", torch.mean(a_bw), global_step=step)
                else: 
                    # mod does not have activation quantizer for float functionals with quant_gemm_only
                    # input_shape is None when act_quant is not used during 
                    # quant_relu_only for residual connections.
                    assert type(mod) in [FloatFunctional, QConvBn2D, QLinear, QAdaptiveAvgPool2d, QAvgPool2d]

            if type(mod) not in [FloatFunctional, QInteractionLayer, QEinsum, QSoftmax, 
                    QEmbedding, QLayerNorm, QAdaptiveAvgPool2d, QAvgPool2d, QuantStub]:
                w_obs = mod.weight_quantizer.observer
                w_elems = torch.prod(torch.tensor(mod.weight_quantizer.input_shape))
                w_sum += w_elems
                w_bw = w_obs.bitwidth_for_penalty
                w_bits += torch.mean(w_bw) * w_elems
                if tb_logger: 
                    tb_logger.add_scalar(f"w_bits/{mod.layer_num}", torch.mean(w_bw), global_step=step)

    return w_bits, w_sum, a_bits, a_sum

def train_imagenet(model, train_loader, test_loader, args, lambd, tb_logger,
        target_emac=None, target_weight_bits=8, target_act_bits=8,
        plot_logdir=None):
    
    num_steps = len(train_loader) * args.epochs if args.train_batches == -1 else min(args.train_batches, len(train_loader))
    num_batches = len(train_loader) if args.train_batches == -1 else min(args.train_batches, len(train_loader))
    num_steps = num_batches * args.epochs 
    print(f"Training for {num_steps} steps, {num_batches} batches per epoch, and {args.epochs} epochs")
    
    loss_ema = None
    acc_ema = None
    best_loss = 100
    best_acc = -1
    
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(total=num_batches, file=sys.stdout, leave=False)
        for it, (image, target, _) in enumerate(train_loader):
            if step > args.train_batches:
                break
            step += 1 

            optimizer.zero_grad()
            image = image.to(args.device)
            target = target.to(args.device)

            output = model(image)
            loss = criterion(output, target)
            tb_logger.add_scalar('train/loss', loss, global_step=step)

            if args.kure:
                kurt_loss, num_weights = kurtosis_loss(model, 0., 0)
                loss = loss + kurt_loss / num_weights
            if args.train_noise:
                emac = compute_noise_regularizer(model)
                if args.constrained_loss:
                    loss = loss + lambd * torch.clamp(torch.log(emac) - np.log(target_emac), min=0.)
                else:
                    loss = loss + lambd * torch.log(emac)
                        
                tb_logger.add_scalar('train/log_energy', torch.log(emac), global_step=step)
                tb_logger.add_scalar('train/energy', emac, global_step=step)
            if args.train_bitwidth:
                w_bits, w_sum, a_bits, a_sum = compute_bit_regularizer(model, 
                        args.round_bitwidth, args.max_act, tb_logger=tb_logger, step=step)
                w_bits /= w_sum 
                a_bits /= a_sum

                if args.constrained_loss:
                    bit_loss = lambd * (torch.clamp(w_bits - target_weight_bits, min=0.)
                                             + torch.clamp(a_bits - target_act_bits, min=0.))
                elif args.constrained_loss_squared:
                    bit_loss = lambd * ((w_bits - target_weight_bits) ** 2.
                                             + (a_bits - target_act_bits) ** 2.)
                else:
                    bit_loss = lambd * (w_bits + a_bits)
                
                loss += bit_loss
                tb_logger.add_scalar('train/bits_act', a_bits, global_step=step)
                tb_logger.add_scalar('train/bits_weight', w_bits, global_step=step)

            tb_logger.add_scalar('train/total_loss', loss, global_step=step)

            acc = accuracy_topk(output, target, topk=(1,))[0].item()
            loss_ema = loss.item() if loss_ema is None else 0.1 * loss.item() + 0.9 * loss_ema
            acc_ema = acc if acc_ema is None else 0.1 * acc + 0.9 * acc_ema
            if args.checkpoint:
                if not args.train_bitwidth or w_bits - target_weight_bits <= 0 and a_bits - target_act_bits <= 0:
                    if acc_ema > best_acc:
                        print("Saving model with acc {0}".format(acc_ema))
                        best_acc = acc_ema
                        torch.save(model.state_dict(), f"{plot_logdir}/model_acc.pth")

            loss.backward()
            optimizer.step()
            loss = loss.item()

            tb_logger.add_scalar('train/acc', acc, global_step=step)
            
            if args.train_noise:
                pbar.set_description("Epoch {:d}, loss {:.4f}, train acc {:.4f}, emac {:.4f}"\
                        .format(epoch, loss_ema, acc_ema, emac))
            elif args.train_bitwidth:
                pbar.set_description("Epoch {:d}, total_loss {:.4f}, "\
                "bit_loss {:.4f}, train acc {:.4f}, w_bits {:.4f}, a_bits "\
                "{:.4f}, target_w {:.4f}, target_a {:.4f}, lambda "\
                "{:0.4f}".format(epoch, loss_ema, bit_loss, acc_ema,
                w_bits, a_bits, target_weight_bits,
                target_act_bits, lambd))
            else:
                pbar.set_description("Epoch {:d}, loss {:.4f}, train acc {:.4f}"\
                          .format(epoch, loss_ema, acc_ema))
            pbar.update(1)
        pbar.close()

    return model


def train_bert(model, train_loader, test_loader, args, lambd, target_emac=None, target_weight_bits=8, target_act_bits=8):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_ema = None
    acc_ema = None
    num_batches = len(train_loader) if args.train_batches == -1 else min(len(train_loader), args.train_batches)

    for epoch in range(args.epochs):
        pbar = tqdm(total=num_batches, file=sys.stdout, leave=False)
        for it, inputs in enumerate(train_loader):
            if it > args.train_batches:
                break

            step = it + 1 + epoch * len(train_loader)
            optimizer.zero_grad()
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)

            output = model(**inputs)
            loss = output["loss"] if isinstance(output, dict) else output[0]

            if args.kure:
                kurt_loss, num_weights = kurtosis_loss(model, 0., 0)
                loss = loss + kurt_loss / num_weights
            if args.train_noise:
                emac = compute_noise_regularizer(model)
                if args.constrained_loss:
                    loss = loss + lambd * torch.clamp(torch.log(emac) - np.log(target_emac), min=0.)
                else:
                    loss = loss + lambd * torch.log(emac)
            if args.train_bitwidth:
                w_bits, w_sum, a_bits, a_sum = compute_bit_regularizer(model, args.round_bitwidth, args.max_act)
                w_bits /= w_sum
                a_bits /= a_sum
                if args.constrained_loss:
                    bit_loss = lambd * (torch.clamp(w_bits - target_weight_bits, min=0.)
                                             + torch.clamp(a_bits - target_act_bits, min=0.))
                else:
                    bit_loss = lambd * (w_bits + a_bits)
                loss += bit_loss

            loss.backward()
            optimizer.step()
            loss = loss.item()
            acc = accuracy_topk(output['logits'], inputs['labels'], topk=(1,))[0].item()

            loss_ema = loss if loss_ema is None else 0.1 * loss + 0.9 * loss_ema
            acc_ema = acc if acc_ema is None else 0.1 * acc + 0.9 * acc_ema

            if args.train_noise:
                pbar.set_description("Epoch {:d}, loss {:.4f}, train acc {:.4f}, emac {:.4f}"\
                        .format(epoch, loss_ema, acc_ema, emac))
            elif args.train_bitwidth:
                pbar.set_description("Epoch {:d}, loss {:.4f}, train acc {:.4f}, w_bits {:.4f}, a_bits {:.4f}"\
                        .format(epoch, loss_ema, acc_ema, w_bits, a_bits))
            else:
                pbar.set_description("Epoch {:d}, loss {:.4f}, train acc {:.4f}"\
                          .format(epoch, loss_ema, acc_ema))
            pbar.update(1)
        pbar.close()


    return model
