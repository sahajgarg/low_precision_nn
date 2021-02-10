for seed in 1 2 3 4 5 
do 
python main.py --act_bits 8 --weight_bits 8 --constrained_loss --target_weight_bits 2.73 2.98 3.23 3.48 3.73 3.98 4.23 --target_act_bits 8 --lambd 10. --lr 0.01 --run_name weight_symmetric_8wva_lowtarget_1024/"$seed" --data_path /mnt/efs --train_bitwidth  --seed $seed --weight_bits_only --round_bitwidth --checkpoint --train_batches 1500 --weight_symmetric --train_subset 1024

python main.py --act_bits 8 --weight_bits 8 --constrained_loss --target_weight_bits 2.5 2.75 3. 3.25 3.5 3.75 4.  --target_act_bits 8 --lambd 10. --lr 0.01 --run_name weight_asymmetric_8wva_lowtarget_1024/"$seed" --data_path /mnt/efs --train_bitwidth  --seed $seed --weight_bits_only --round_bitwidth --checkpoint --train_batches 1500 --train_subset 1024
done

