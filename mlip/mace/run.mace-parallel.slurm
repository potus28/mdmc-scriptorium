#!/bin/bash
#SBATCH --job-name=mace
#SBATCH --account=rg2
#SBATCH --reservation=h100-testing
#SBATCH --partition=gpu-h100
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:4
#SBATCH --output=slurm.%j
#SBATCH --error=slurm.%j
#SBATCH --exclusive

source /nopt/nrel/apps/env.sh
module load anaconda3
conda activate mace

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0


for i in  N2_v2_r5_l1_f128  N2_v3_r5_l0_f256  N2_v3_r6_l1_f128 N2_v3_r5_l0_f128 ; do
	cd $i
	bash run.mace.sh $CUDA_VISIBLE_DEVICES > /dev/null 2> /dev/null &
	export CUDA_VISIBLE_DEVICES=$(($CUDA_VISIBLE_DEVICES + 1 ))
	cd ..
done
wait

cd N2_v3_r5_l1_f128
bash run.mace.sh $CUDA_VISIBLE_DEVICES > /dev/null 2> /dev/null 
cd ..
