#!/bin/bash
#SBATCH --partition=batchall
#SBATCH --time=4800000:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=wat-pureSi-aimd
#SBATCH --output=slurm.out.%j
#SBATCH --error=slurm.error.%j

. "/home/wwilson/Programs/miniconda3/etc/profile.d/conda.sh"

source /home/wwilson/Programs/cp2k/2024.1/tools/toolchain/install/setup
export PATH=$PATH:/home/wwilson/Programs/cp2k/2024.1/tools/toolchain/install/mpich-4.0.3/bin/
export PATH=$PATH:/home/wwilson/Programs/cp2k/2024.1/exe/local/


export OMP_NUM_THREADS=1 
export OMP_PROC_BIND=true
export OMP_PLACES=cores

export ASE_CP2K_COMMAND="/home/wwilson/Programs/cp2k/2024.1/tools/toolchain/install/mpich-4.0.3/bin/mpiexec -bind-to core -n 32 /home/wwilson/Programs/cp2k/2024.1/exe/local/cp2k_shell.psmp"



echo "START: $(date)"

if [ -f cp2k-md-1.restart ]; then
    /home/wwilson/Programs/cp2k/2024.1/tools/toolchain/install/mpich-4.0.3/bin/mpiexec -bind-to core -n 32 /home/wwilson/Programs/cp2k/2024.1/exe/local/cp2k.psmp -i cp2k-md-1.restart -o cp2k.out
else
    /home/wwilson/Programs/cp2k/2024.1/tools/toolchain/install/mpich-4.0.3/bin/mpiexec -bind-to core -n 32 /home/wwilson/Programs/cp2k/2024.1/exe/local/cp2k.psmp -i cp2k.inp -o cp2k.out
fi


echo "END: $(date)"
