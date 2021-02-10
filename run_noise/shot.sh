for model in resnet50  
do 
python main.py --act_bits -1 --weight_bits -1 --e_mac 10000 --noise_type shot --run_name "$model""_train_constrained" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 2.0 --model $model --target_emac  10. 20. 50. 100. 200. 500. 1000. 2000. 5000. 10000. --constrained_loss  

python main.py --act_bits -1 --weight_bits -1 -e_mac 10000 --noise_type shot --run_name "$model""_train_constrained_per_channel" --data_path /mnt/efs --train_noise --lr 0.01 --eval_batches 500 --lambd 2.0 --model $model --target_emac  10. 20. 50. 100. 200. 500. 1000. 2000. 5000. 10000. --constrained_loss --noise_per_channel

python main.py --act_bits -1 --weight_bits -1 --e_mac 10. 20. 50. 100. 200. 500. 1000. 2000. 5000. 10000. --noise_type shot --run_name "$model""_vanilla" --data_path /mnt/efs  --eval_batches 500  --model $model
done
