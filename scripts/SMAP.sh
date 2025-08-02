export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# python main.py --anormly_ratio 0.2 --num_epochs 10  --batch_size 128  --win_size 105   --mode train --dataset SMAP  --data_path dataset/SMAP --input_c 25    --output_c 25


export CUDA_VISIBLE_DEVICES=7
python main.py --anormly_ratio 0.8  --num_epochs 10        --batch_size 32  --win_size 105      --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25





