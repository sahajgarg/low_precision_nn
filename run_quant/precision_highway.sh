python main.py --weight_bits 8 --quant_gemm_only --act_bits 3. 4. 5. 6. 7. 8. --run_name gemm_only_8wva_percentile --data_path /mnt/efs  --act_observer percentile --percentile 99.99 --calibration_batches 10 --ignore first+last   

python main.py --weight_bits 8 --act_bits 3. 4. 5. 6. 7. 8. --run_name quant_all_8wva_percentile --data_path /mnt/efs --act_observer percentile --percentile 99.99 --calibration_batches 10 --ignore first+last  

python main.py --weight_bits 8 --act_bits 3. 4. 5. 6. 7. 8. --run_name relu_only_8wva_percentile --data_path /mnt/efs --quant_relu_only --act_observer percentile --percentile 99.99 --calibration_batches 10 --ignore first+last    
