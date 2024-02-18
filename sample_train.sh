CUDA_VISIBLE_DEVICES=0 python train.py --logging \
--epochs 18 \
--batch_size 128 \
--lr 0.001 \
--embedding_dim 300 \
--max_length 300 \
--model NBoW