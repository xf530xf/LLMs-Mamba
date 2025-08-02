


# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# python main.py --anormly_ratio 1 --num_epochs 50    --batch_size 128  --win_size 150 --mode train --dataset PSM  --data_path dataset/PSM --input_c 25    --output_c 25



export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
python main.py --anormly_ratio 2  --num_epochs 10       --batch_size 128  --win_size 150      --mode test    --dataset PSM   --data_path dataset/PSM  --input_c 25    --output_c 25


