# Adaptation of the pytorchcv resnet for quantization, including 
# layer fusion and float functionals for residual connections.

import torch
import torch.nn as nn
import pytorchcv.model_provider as pytcv
import pytorchcv
import os

class QResBlock(pytcv.ResBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bias=False,
                 use_bn=True):
        super(QResBlock, self).__init__(in_channels, out_channels, stride, bias, use_bn)

    def fuse_model(self):
        torch.quantization.fuse_modules(self.conv1, ['conv', 'bn', 'activ'], inplace=True)
        torch.quantization.fuse_modules(self.conv2, ['conv', 'bn'], inplace=True)

class QResBottleneck(pytcv.ResBottleneck):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding=1,
                 dilation=1,
                 conv1_stride=False,
                 bottleneck_factor=4):
        super(QResBottleneck, self).__init__(in_channels, out_channels, stride,
                padding, dilation, conv1_stride, bottleneck_factor)

    def fuse_model(self):
        torch.quantization.fuse_modules(self.conv1, ['conv', 'bn', 'activ'], inplace=True)
        torch.quantization.fuse_modules(self.conv2, ['conv', 'bn', 'activ'], inplace=True)
        torch.quantization.fuse_modules(self.conv3, ['conv', 'bn'], inplace=True)

class QResUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding=1,
                 dilation=1,
                 bias=False,
                 use_bn=True,
                 bottleneck=True,
                 conv1_stride=False):
        super(QResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = QResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                conv1_stride=conv1_stride)
        else:
            self.body = QResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                use_bn=use_bn)
        if self.resize_identity:
            self.identity_conv = pytorchcv.models.common.conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                use_bn=use_bn,
                activation=None)
        self.skip = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.skip.add_relu(x, identity)
        return x

    def fuse_model(self):
        if self.resize_identity:
            torch.quantization.fuse_modules(self.identity_conv, ['conv', 'bn'], inplace=True)
        for m in self.modules():
            if type(m) in [QResBlock, QResBottleneck]:
                m.fuse_model()
    
class QResInitBlock(pytcv.ResInitBlock):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(QResInitBlock, self).__init__(in_channels, out_channels)
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self.conv, ['conv', 'bn', 'activ'], inplace=True)


class QResNet(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(QResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", QResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), QResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) in [QResUnit, QResInitBlock]:
                m.fuse_model()

def qresnet50_ptcv(**kwargs):
    return get_qresnet(blocks=50, model_name="resnet50", **kwargs)

def get_qresnet(blocks,
               bottleneck=None,
               conv1_stride=True,
               width_scale=1.0,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = QResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from pytorchcv.models.model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net




