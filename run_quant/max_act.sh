python main.py --act_bits 3. 4. 5. 6. 7. 8. --weight_bits 8 --run_name baseline_8wva_ignore --data_path /mnt/efs --seed 8  --act_observer percentile --percentile 99.99 --calibration_batches 10 --ignore last

python main.py --act_bits 8 --weight_bits 8 --max_act_analytical --target_act_bits 3. 4. 5. 6. 7. 8. --run_name max_act_8wva_ignore --data_path /mnt/efs --seed 8  --act_observer percentile --percentile 99.99 --calibration_batches 10 --ignore last
