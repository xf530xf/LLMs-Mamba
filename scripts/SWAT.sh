# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
# # python main.py --anormly_ratio 0.5 --num_epochs 20   --batch_size 256 --win_size 100    --mode train --dataset SWAT  --data_path dataset/SWAT   --input_c 51  --output_c 51


export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256 --win_size 100   --mode test    --dataset SWAT   --data_path dataset/SWAT     --input_c 51 --output_c 51



# pa_accuracy           : 0.9918
# pa_precision          : 0.9369
# pa_recall             : 1.0000
# pa_f_score            : 0.9674
# MCC_score             : nan
# Affiliation precision : 0.5504
# Affiliation recall    : 0.9798
# R_AUC_ROC             : 0.9826
# R_AUC_PR              : 0.9574
# VUS_ROC               : 0.9829