noise_type="thermal"
bits=8
mntefs

models=("resnet50" "googlenet" "shufflenetv2" "inceptionv3" "mobilenet")
target_acc=(74.012 67.744 67.402 75.25 68.31)

for i in 0 1 3 
do
    model=${models[$i]}
    python main.py --act_bits $bits --weight_bits $bits --e_mac 10000 --noise_type $noise_type --run_name "$model""_train_binary_search""$noise_type" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 8.0 --model $model --binary_search_acc ${target_acc[$i]} --search_min_emac 5. --search_max_emac 500. --constrained_loss --train_batches 1500 --act_observer percentile --percentile 99.99 --calibration_batches 5

    python main.py --act_bits $bits --weight_bits $bits --e_mac 10000 --noise_type $noise_type --run_name "$model""_train_channel_binary_search""$noise_type" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 8.0 --model $model --binary_search_acc ${target_acc[$i]} --search_min_emac 5. --search_max_emac 500. --constrained_loss   --noise_per_channel --train_batches 1500 --act_observer percentile --percentile 99.99 --calibration_batches 5

    python main.py --act_bits $bits --weight_bits $bits --e_mac 10000 --noise_type $noise_type --run_name "$model""_vanilla_binary_search""$noise_type" --data_path /mnt/efs  --eval_batches 500 --lambd 2.0 --model $model --binary_search_acc ${target_acc[$i]} --search_min_emac 10. --search_max_emac 1000.  --train_batches 1500 --act_observer percentile --percentile 99.99 --calibration_batches 5
done

for i in 2 4
do
    model=${models[$i]}
    python main.py --act_bits $bits --weight_bits $bits --e_mac 10000 --noise_type $noise_type --run_name "$model""_train_binary_search""$noise_type" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 8.0 --model $model --binary_search_acc ${target_acc[$i]} --search_min_emac 20. --search_max_emac 2000. --constrained_loss --train_batches 1500 --act_observer percentile --percentile 99.99 --calibration_batches 5

    python main.py --act_bits $bits --weight_bits $bits --e_mac 10000 --noise_type $noise_type --run_name "$model""_train_channel_binary_search""$noise_type" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 8.0 --model $model --binary_search_acc ${target_acc[$i]} --search_min_emac 20. --search_max_emac 2000. --constrained_loss   --noise_per_channel --train_batches 1500 --act_observer percentile --percentile 99.99 --calibration_batches 5

    python main.py --act_bits $bits --weight_bits $bits --e_mac 10000 --noise_type $noise_type --run_name "$model""_vanilla_binary_search""$noise_type" --data_path /mnt/efs  --eval_batches 500 --lambd 2.0 --model $model --binary_search_acc ${target_acc[$i]} --search_min_emac 50. --search_max_emac 5000.  --train_batches 1500 --act_observer percentile --percentile 99.99 --calibration_batches 5
done
