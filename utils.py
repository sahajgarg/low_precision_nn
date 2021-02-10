import os
import sys
import torchvision
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.quantized as nnq
from transformers import AutoConfig, AutoModelForSequenceClassification
from quantized_modules import *
from noise_models import *
import models
import models.ptcv as ptcv
from shutil import copyfile

from models.dlrm.dlrm_s_pytorch import DLRM_Net
from models.dlrm.interaction_layer import InteractionLayer

from models.bert import Einsum, BertForSequenceClassification

MODULE_DICT = {nn.Linear: QLinear,
               nni.LinearReLU: QLinearReLU,
               nni.ConvBn2d: QConvBn2D,
               nni.ConvBnReLU2d: QConvBnReLU2d,
               nni.ConvReLU2d: QConvReLU2d,
               nnq.FloatFunctional: FloatFunctional,
               torch.quantization.QuantStub: QuantStub,
               nn.Embedding: QEmbedding,
               nn.EmbeddingBag: QEmbeddingBag,
               InteractionLayer: QInteractionLayer,
               nn.LayerNorm: QLayerNorm,
               nn.Softmax: QSoftmax,
               Einsum: QEinsum,
               nn.AdaptiveAvgPool2d: QAdaptiveAvgPool2d, 
               nn.AvgPool2d: QAvgPool2d
               }


class Aggregator(object):
    def __init__(self):
        self.preds = None
        self.targets = None

    def reset(self):
        self.preds = None
        self.targets = None

    def update(self, preds, targets):
        preds_np = preds.detach().cpu().numpy().squeeze()
        targets_np = targets.detach().cpu().numpy().squeeze()
        if self.preds is None:
            self.preds = preds_np
            self.targets = targets_np
        else:
            self.preds = np.concatenate([self.preds, preds_np])
            self.targets = np.concatenate([self.targets, targets_np])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy_binary(preds, targets):
    preds_np = preds.detach().cpu().numpy()  # numpy array
    targets_np = targets.detach().cpu().numpy()  # numpy array
    batch_size = targets.shape[0]
    num_correct = np.sum((np.round(preds_np, 0) == targets_np).astype(np.uint8))

    return num_correct / batch_size


def save_outputs(model, data_loader, neval_batches, device, quantizer, save_path):
    model.eval()
    cnt = 0
    pred_to_class = {}
    
    with torch.no_grad():
        for _, targets, paths in data_loader:
            for path, target in zip(paths, targets) :
                path = os.path.normpath(path)
                class_dir = path.split(os.sep)[-2]
                pred_to_class[target.item()] = class_dir
            if len(pred_to_class) == 1000:
                break
        
        pbar = tqdm(total=min(len(data_loader), neval_batches), file=sys.stdout, leave=False)
        for image, _, image_paths in data_loader:
            image = image.to(device)
            target = target.to(device)
            image_quantized = quantizer(image)
            output = model(image_quantized)
            cnt += 1
            
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t().squeeze()

            for path, model_pred, in zip(image_paths, pred.squeeze()):
                dir_name = pred_to_class[model_pred.item()]
                dir_path = os.path.join(save_path, dir_name)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                new_file_path = os.path.join(dir_path, os.path.basename(path))
                copyfile(path, new_file_path)
            
            pbar.update(1)
            if cnt >= neval_batches > 0:
                pbar.close()
        pbar.close()


def accuracy_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
       
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 

def load_model_quantized(model_name, device, dataset, num_labels):
    pretrained = (dataset == "imagenet")
    if model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=pretrained, progress=True, quantize=False)
    elif model_name == "resnet50":
        model = torchvision.models.quantization.resnet50(pretrained=pretrained, progress=True, quantize=False)
    elif model_name == "resnet50_ptcv":
        model = ptcv.qresnet50_ptcv(pretrained=pretrained)
    elif model_name == "inceptionv3":
        model = models.inception_v3(pretrained=pretrained, progress=True, quantize=False)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=pretrained, progress=True, quantize=False)
    elif model_name == "shufflenetv2":
        model = models.shufflenet_v2_x1_0(pretrained=pretrained, progress=True, quantize=False)
    elif model_name == 'dlrm':
        # These arguments are hardcoded to the defaults from DLRM (matching the pretrained model).
        model = DLRM_Net(16,
                         np.array([1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
                                   8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547,
                                   18, 15, 286181, 105, 142572], dtype=np.int32),
                         np.array([13, 512, 256,  64,  16]),
                         np.array([367, 512, 256,   1]),
                         'dot', False, -1, 2, True, 0., 1, False, 'mult', 4, 200, False, 200)
        ld_model = torch.load('data/dlrm.pt')
        model.load_state_dict(ld_model["state_dict"])
    elif model_name == 'bert':
        config = AutoConfig.from_pretrained(
            'bert-base-cased',
            num_labels=num_labels,
            finetuning_task='mnli',
        )
        model = BertForSequenceClassification.from_pretrained('data/bert.bin', from_tf=False, config=config)
    else:
        raise ValueError("Unsupported model type")

    if dataset == "cifar10":
        ld_model = torch.load(f"data/{model_name}.pt")
        model.load_state_dict(ld_model)

    model = model.to(device)
    return model


def load_model(model_name, device, dataset, num_labels):
    pretrained = (dataset == "imagenet")
    if model_name == "mobilenet":
        model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrained)
    elif model_name == "inceptionv3":
        model = torchvision.models.inception_v3(pretrained=pretrained)
    elif model_name == "googlenet":
        model = torchvision.models.googlenet(pretrained=pretrained)
    elif model_name == "shufflenetv2":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
    elif model_name == 'dlrm':
        # These arguments are hardcoded to the defaults from DLRM (matching the pretrained model).
        model = DLRM_Net(16,
                         np.array([1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
                                   8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547,
                                   18, 15, 286181, 105, 142572], dtype=np.int32),
                         np.array([13, 512, 256,  64,  16]),
                         np.array([367, 512, 256,   1]),
                         'dot', False, -1, 2, True, 0., 1, False, 'mult', 4, 200, False, 200)
        ld_model = torch.load('data/dlrm.pt')
        model.load_state_dict(ld_model["state_dict"])
    elif model_name == 'bert':
        config = AutoConfig.from_pretrained(
            'bert-base-cased',
            num_labels=num_labels,
            finetuning_task='mnli',
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            'data/bert.bin',
            from_tf=False,
            config=config,
        )
    else:
        raise ValueError("Unsupported model type")

    if dataset == "cifar10":
        ld_model = torch.load(f"data/{model_name}.pt")
        model.load_state_dict(ld_model)

    model = model.to(device)
    return model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def std(x, dim=None):
    mean = x.mean() if dim is None else x.mean(dim=dim, keepdim=True)
    sum_sq_diff = torch.mean(torch.square(x - mean)) if dim is None else torch.mean(torch.square(x - mean), dim=dim)
    return torch.sqrt(sum_sq_diff + 1e-8)

layer = 0
def set_layers(module):
    for name, mod in module.named_children():
        if type(mod) not in MODULE_DICT.values():
            set_layers(mod)
        else:
            global layer
            if type(mod) == FloatFunctional:
                mod.layer_num = f"{layer}_residual"
            else:
                layer += 1
                mod.layer_num = f"{layer}"

def start_recorders(module):
    for name, mod in module.named_children():
        if isinstance(mod, Gaussian):
            mod.recording = True
        else:
            start_recorders(mod)

def stop_recorders(module):
    for name, mod in module.named_children():
        if isinstance(mod, Gaussian):
            mod.recording = False
        else:
            stop_recorders(mod)

def start_recording_clean(module):
    for name, mod in module.named_children():
        if isinstance(mod, Gaussian):
            mod.recording_clean = True
        else:
            start_recording_clean(mod)

def stop_recording_clean(module):
    for name, mod in module.named_children():
        if isinstance(mod, Gaussian):
            mod.recording_clean = False
        else:
            stop_recording_clean(mod)

def reset_stats(module):
    for name, mod in module.named_children():
        if isinstance(mod, Gaussian):
            mod.stats = None
            mod.n = 0
        else:
            reset_stats(mod)

def relax_quantization(module):
    for name, mod in module.named_children():
        if type(mod) not in MODULE_DICT.values():
            relax_quantization(mod)
        else:
            if hasattr(mod, "activation_quantizer"):
                mod.activation_quantizer.approximate_quantize = True
            if hasattr(mod, "weight_quantizer"):
                mod.weight_quantizer.approximate_quantize = True

def stop_noise(module):
    for name, mod in module.named_children():
        if isinstance(mod, Gaussian):
            mod.add_noise = False
        else:
            stop_noise(mod)


def start_noise(module):
    for name, mod in module.named_children():
        if isinstance(mod, Gaussian):
            mod.add_noise = True
        else:
            start_noise(mod)

def finalize_bitwidth_observer(obs):
    if obs.train_via_scale:
        bw = torch.log2(torch.abs(obs.max_val -
            obs.min_val) / torch.exp(obs.log_scale) + 1.)
    else:
        bw = 2. ** obs.log_bitwidth.data

    if not obs.discrete_bitwidth:
        bw = torch.round(bw)

    if obs.train_via_scale:
        obs.log_scale.data = torch.log(torch.abs(obs.max_val - obs.min_val) / (2. ** bw - 1.))
    else:
        obs.log_bitwidth.data = torch.log2(bw)

def finalize_bitwidth(module):
    for name, mod in module.named_children():
        if type(mod) not in MODULE_DICT.values():
            finalize_bitwidth(mod)
        else:
            mod.activation_quantizer.approximate_quantize = False
            finalize_bitwidth_observer(mod.activation_quantizer.observer)
            if not isinstance(mod, FloatFunctional):
                mod.weight_quantizer.approximate_quantize = False
                finalize_bitwidth_observer(mod.weight_quantizer.observer)

def print_modules(module, indent=""):
    for name, mod in module.named_children():
        print(f"{indent}{name}\t{type(mod)}")
        if type(mod) not in MODULE_DICT.keys() and type(mod) not in MODULE_DICT.values():
            print_modules(mod, f"{indent}\t")
