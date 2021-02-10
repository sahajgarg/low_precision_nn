for seed in 1 2 3 4 5 
do 
	python main.py --act_bits 8 --weight_bits 8 --constrained_loss --target_act_bits 8. --target_weight_bits 3. 4. 5. 6. 7. 8 --lambd 2. --lr 0.01 --run_name vw8a_generated_labels/"$seed" --data_path /mnt/efs --train_bitwidth --seed $seed --weight_bits_only --round_bitwidth --checkpoint --train_batches 1500 --train_data_path /mnt/efs/non_quant_model_output --val_data_path /mnt/efs/full_imagenet/val_2012/val_formatted --train_subset 1024

	python main.py --act_bits 8 --weight_bits 8 --constrained_loss --target_act_bits 8. --target_weight_bits 3. 4. 5. 6. 7. 8 --lambd 2. --lr 0.01 --run_name vw8a_real_labels/"$seed" --data_path /mnt/efs --train_bitwidth --seed $seed --weight_bits_only --round_bitwidth --checkpoint --train_batches 1500 --train_subset 1024
done
