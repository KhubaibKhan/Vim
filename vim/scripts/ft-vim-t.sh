#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vim_kh
cd /data_hdd1/users/khubaib/mamba_unders/Vim/vim;

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 128 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --num_workers 25 --data-path imagenet-1k --output_dir ./output/vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --epochs 30 --finetune <path_to_pt_ckpt> --no_amp
