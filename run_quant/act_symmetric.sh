python main.py --weight_bits 8 --act_bits 3. 4. 5. 6. 7. 8. --run_name percentile_symmetric_8wva --data_path /mnt/efs --seed 8 --quant_relu_only --act_symmetric --act_observer percentile --percentile 99.99 --calibration_batches 10 --ignore first+last

python main.py --weight_bits 8 --act_bits 2. 3. 4. 5. 6. 7. --run_name percentile_asymmetric_8wva --data_path /mnt/efs --seed 8 --quant_relu_only  --act_observer percentile --percentile 99.99 --calibration_batches 10 --ignore first+last
