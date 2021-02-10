for model in resnet50 googlenet shufflenetv2 inceptionv3 mobilenet
do
python main.py --act_bits -1 --weight_bits -1 --run_name "$model""fp_baseline" --data_path /mnt/efs  --eval_batches 500 --model $model 

python main.py --act_bits 8 --weight_bits 8 --run_name "$model""8b_baseline" --data_path /mnt/efs  --eval_batches 500 --model $model 

python main.py --act_bits 8 --weight_bits 8 --run_name "$model""percentile_baseline" --data_path /mnt/efs  --eval_batches 500 --model $model  --act_observer percentile --percentile 99.99 --calibration_batches 5
done 
