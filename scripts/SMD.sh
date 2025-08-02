# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
# python main.py --anormly_ratio 0.5 --num_epochs 30   --batch_size 32 --win_size 100  --mode train --dataset SMD  --data_path dataset/SMD   --input_c 38 




export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
python main.py --anormly_ratio 0.3 --num_epochs 10   --batch_size 32 --win_size 100 --mode test    --dataset SMD   --data_path dataset/SMD     --input_c 38  --output_c 38


 

