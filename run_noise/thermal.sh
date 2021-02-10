for model in resnet50  
do 
python main.py --act_bits 8 --weight_bits 8 --e_mac 1000 --noise_type thermal --run_name "$model""_train_thermal_percentile" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 4.0 --model $model --target_emac 1. 2. 5. 10. 20. 50. 100. 200. 500. 1000. --constrained_loss   --train_batches 1500 --act_observer percentile --percentile 99.99 --calibration_batches 5

python main.py --act_bits 8 --weight_bits 8 --e_mac 1000 --noise_type thermal --run_name "$model""_train_per_channel_thermal_percentile" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 4.0 --model $model --target_emac 1. 2. 5. 10. 20. 50. 100. 200. 500. 1000. --constrained_loss --noise_per_channel --train_batches 1500 --act_observer percentile --percentile 99.99 --calibration_batches 5


python main.py --act_bits 8 --weight_bits 8 --e_mac 1. 2. 5. 10. 20. 50. 100. 200. 500. 1000. --noise_type thermal --run_name "$model""_vanilla_thermal_percentile" --data_path /mnt/efs  --eval_batches 500  --model $model --train_batches 1500 --act_observer percentile --percentile 99.99 --calibration_batches 5

done
