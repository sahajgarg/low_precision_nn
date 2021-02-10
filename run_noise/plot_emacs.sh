 python main.py --act_bits -1 --weight_bits -1 --e_mac 1000 --noise_type shot --run_name resnet_plot --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 2.0 --model resnet50 --target_emac 25. --constrained_loss --train_batches 1500

python main.py --act_bits -1 --weight_bits -1 --e_mac 1000 --noise_type shot --run_name mobilenet_plot --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 2.0 --model mobilenet --target_emac 163.3 --constrained_loss --train_batches 1500

python main.py --act_bits -1 --weight_bits -1 --e_mac 20 --noise_type shot --run_name bert_plot --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 2.0 --model bert --target_emac 12.5 --constrained_loss --dataset mnli --train_batches 1500
