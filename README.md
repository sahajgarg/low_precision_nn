# Post-Training Mixed-Precision Quantization

This repository contains a pytorch implementation of the two papers [Dynamic Precision Analog Computing for Neural Networks](https://arxiv.org/abs/2102.06365) and [Confounding Tradeoffs for Neural Network Quantization](https://arxiv.org/abs/2102.06366). These works explore quantization to low bit precision and in the presence of analog noise, and evaluate quantization of neural networks at mixed precision for different layers or channels of neural networks.  

The repository contains implementations of the following models for simulated quantization:
- Resnet50
- BERT
- DLRM 
- InceptionV3
- ShufflenetV2
- MobilenetV2
- Googlenet

## Setup 

We use the Amazon EC2 Linux Deep Learning AMI with the pytorch\_latest\_p37 conda environment with the following additional packages: 

```
pip install transformers datasets pytorchcv tensorboard tensorboardX
pip install --upgrade seaborn
```

If you are running without the EC2 AMI, you may need to install the following packages:
```
pip install pytorch torchvision numpy pandas matplotlib
```

Mount imagenet to your favorite location, and you should be good to go! 

## Usage
Scripts for generating the results are in the `run_noise` and `run_quant` directory for [Dynamic Precision Analog Computing for Neural Networks](https://arxiv.org/abs/2102.06365) and [Confounding Tradeoffs for Neural Network Quantization](https://arxiv.org/abs/2102.06366), respectively. You can generate the results using:

```
source run_noise/plot_emacs.sh
source run_quant/per_channel.sh
...
```

For example, to train mixed precision energy allocations for analog computing subject to thermal noise, you could execute the following command:

```
python main.py \
    --act_bits 8 \
    --weight_bits 8 \
    --e_mac 1000 \
    --noise_type thermal \
    --run_name thermal_test \
    --data_path /mnt/efs \
    --train_noise --lr 0.01 \
    --eval_batches 500 \
    --lambd 8.0 \
    --model resnet50 \
    --target_emac 1. 2. 5. 10. 20. 50. 100. 200. 500. 1000. \
    --constrained_loss   \
    --train_batches 1500 \
    --act_observer percentile \
    --percentile 99.99 \
    --calibration_batches 5
```

To train mixed precision bitwidths for digital computing, you could execute the following command: 

```
python main.py \
    --train_bitwidth \
    --model resnet50 \
    --lambd 2.0 \
    --lr 0.01 \
    --data_path /mnt/efs \
    --weight_bits 8. \
    --act_bits 8. \
    --run_name train_8A4W \
    --constrained_loss \
    --target_weight_bits 4. \
    --target_act_bits 8. \
    --train_qminmax  \
    --checkpoint \
    --round_bitwidth \
    --train_batches 1500 \
    --weight_bits_only \
    --train_subset 1024
```

## References
If you find the idea or code useful for your research, please consider citing our work:

```
@misc{garg2021dynamic,
      title={Dynamic Precision Analog Computing for Neural Networks}, 
      author={Sahaj Garg and Joe Lou and Anirudh Jain and Mitchell Nahmias},
      year={2021},
      eprint={2102.06365},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{garg2021confounding,
      title={Confounding Tradeoffs for Neural Network Quantization}, 
      author={Sahaj Garg and Anirudh Jain and Joe Lou and Mitchell Nahmias},
      year={2021},
      eprint={2102.06366},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
