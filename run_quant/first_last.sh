for seed in 1 2 3 4 5
do 
python main.py --train_bitwidth --model resnet50 --lambd 2.0 --lr 0.01 --data_path /mnt/efs --weight_bits 8. --act_bits 8. --eval_batches 500 --run_name include_first_last_8A4W_1024/"$seed" --constrained_loss --target_weight_bits 4. --target_act_bits 8. --train_qminmax  --checkpoint --seed $seed --round_bitwidth --train_batches 1500 --weight_bits_only --train_subset 1024

python main.py --train_bitwidth --model resnet50 --lambd 2.0 --lr 0.01 --data_path /mnt/efs --weight_bits 8. --act_bits 8. --eval_batches 500 --run_name first_last_4A4W_1024/"$seed" --constrained_loss --target_weight_bits 4. --target_act_bits 4. --train_qminmax  --checkpoint --seed $seed --round_bitwidth --train_batches 1500 --train_subset 1024

python main.py --train_bitwidth --model resnet50 --lambd 2.0 --lr 0.01 --data_path /mnt/efs --weight_bits 8. --act_bits 8. --eval_batches 500 --run_name first_last_8A4.3W_ignore_1024/"$seed" --constrained_loss --target_weight_bits 4.322694 --target_act_bits 8. --train_qminmax  --checkpoint --seed $seed --round_bitwidth --train_batches 1500 --weight_bits_only --ignore first+last --train_subset 1024

python main.py --train_bitwidth --model resnet50 --lambd 2.0 --lr 0.01 --data_path /mnt/efs --weight_bits 8. --act_bits 8. --eval_batches 500 --run_name first_last_4.2A4.3W_ignore_1024/"$seed" --constrained_loss --target_weight_bits 4.322694 --target_act_bits 4.193291 --train_qminmax  --checkpoint --seed $seed --round_bitwidth --train_batches 1500 --ignore first+last --train_subset 1024
done
