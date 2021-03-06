noise_type="shot"
bits=-1
mntefs

models=("resnet50" "googlenet" "shufflenetv2" "inceptionv3" "mobilenet")
target_acc=(74.012 67.744 67.402 75.25 69.85)

for i in 0 1 2 3 4
do
    model=${models[$i]}
    python main.py --act_bits $bits --weight_bits $bits --e_mac 10000 --noise_type $noise_type --run_name "$model""_train_binary_search""$noise_type" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 2.0 --model $model --binary_search_acc ${target_acc[$i]} --search_min_power 10. --search_max_power 500. --constrained_loss --train_batches 1500 

    python main.py --act_bits $bits --weight_bits $bits --e_mac 10000 --noise_type $noise_type --run_name "$model""_train_channel_binary_search""$noise_type" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 2.0 --model $model --binary_search_acc ${target_acc[$i]} --search_min_power 5 --search_max_power 250. --constrained_loss   --noise_per_channel --train_batches 1500 

    python main.py --act_bits $bits --weight_bits $bits --e_mac 10000 --noise_type $noise_type --run_name "$model""_vanilla_binary_search""$noise_type" --data_path /mnt/efs  --eval_batches 500 --lambd 2.0 --model $model --binary_search_acc ${target_acc[$i]} --search_min_power 100. --search_max_power 1000.  --train_batches 1500 
done
