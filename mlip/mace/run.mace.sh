#!/bin/bash

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=$1
#export CUDA_VISIBLE_DEVICES=$1,1,2,3

nlayers=2
rc=6
lmax=1
body_order=3
nfeat=128
device=cuda  # cpu or cuda
nepoch=50
nswa=35
rns=$2

mace_run_train \
    --name="mace-$rns" \
    --train_file="../train.xyz" \
    --valid_fraction=0.05 \
    --test_file="../test.xyz" \
    --E0s="isolated" \
    --energy_key="energy" \
    --forces_key="forces" \
    --stress_key="stress" \
    --compute_stress=True \
    --model="MACE" \
    --num_interactions=$nlayers \
    --num_channels=$nfeat \
    --max_L=$lmax \
    --correlation=$body_order \
    --forces_weight=1000 \
    --energy_weight=10 \
    --r_max=$rc \
    --batch_size=5 \
    --valid_batch_size=5 \
    --max_num_epochs=$nepoch \
    --start_swa=$nswa \
    --scheduler_patience=5 \
    --patience=15 \
    --eval_interval=1 \
    --ema \
    --swa \
    --swa_forces_weight=10 \
    --error_table="PerAtomRMSEstressvirials" \
    --default_dtype="float32" \
    --device=$device \
    --seed=$rns \
    --restart_latest \
    --save_cpu
