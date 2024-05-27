#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate vim_kh
cd /data_hdd1/users/khubaib/mamba_unders/Vim/vim;

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 256 --drop-path 0.0 --weight-decay 0.1 --num_workers 7 --data-path imagenet-1k --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp --resume /data_hdd1/users/khubaib/mamba_unders/Vim/vim/output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/checkpoint.pth
