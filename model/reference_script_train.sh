
###########################################
# train, CUDA_VISIBLE_DEVIECES=0
# ELMo original 5.5B: batch 20
###########################################

CUDA_VISIBLE_DEVICES=0  python train.py --batch_size 20 --hop 4 --sdim 200 --dr 0.7 --dr_rnn 0.7 --num_train_steps 100000 --graph_prefix 'hotpot' --corpus 'hotpot' --is_save 0 --lr 0.001




###########################################
# train, CUDA_VISIBLE_DEVIECES=0
# ELMo small: batch 80
###########################################

CUDA_VISIBLE_DEVICES=0  python train.py --batch_size 80 --hop 4 --sdim 200 --dr 0.7 --dr_rnn 0.7 --num_train_steps 100000 --graph_prefix 'hotpot_S' --corpus 'hotpot_small' --is_save 0 --lr 0.001