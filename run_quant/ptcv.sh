python main.py --act_bits 8 --weight_bits 3. 4. 5. 6. 7. 8. --run_name vw8a_per_channel_quant_ptcv_uniform --data_path /mnt/efs --seed 1  --model resnet50_ptcv --ignore first+last --train_subset 1024 --calibration_batches 32

python main.py --act_bits 8 --weight_bits 3. 4. 5. 6. 7. 8. --run_name vw8a_per_channel_quant_uniform --data_path /mnt/efs --seed 1  --model resnet50 --ignore first+last --train_subset 1024 --calibration_batches 32

