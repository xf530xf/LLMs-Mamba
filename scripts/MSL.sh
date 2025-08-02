# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7


# python main.py --anormly_ratio 0.85 --num_epochs 10   --batch_size 256  --win_size 100  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55


export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
python main.py --anormly_ratio 0.85  --num_epochs 10      --batch_size 32  --win_size 100   --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55




