export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
python main.py --anormly_ratio 0.5 --num_epochs 20   --batch_size 32 --win_size 100    --mode train --dataset Ettm1  --data_path dataset/SWAT   --input_c 51  --output_c 51


# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
# python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256 --win_size 100   --mode test    --dataset SWAT   --data_path dataset/SWAT     --input_c 51 --output_c 51
