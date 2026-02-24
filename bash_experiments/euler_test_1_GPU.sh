#!/bin/bash
#SBATCH -A es_schin
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:10G
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time 1:00:00
#SBATCH -o job_output/test_rc_RSITMD_distill%j.out
#SBATCH -e job_output/test_rc_RSITMD__distill%j.err

echo "=== Job starting on $(hostname) at $(date) ==="
echo "=== SLURM_PROCID: $SLURM_PROCID, SLURM_LOCALID: $SLURM_LOCALID, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES ==="

module eth_proxy load stack/2024-05 gcc/13.2.0 cuda/12.1.1 python/3.11.6_cuda

source ~/euler_env/bin/activate
echo "Activated Python venv: $(which python)"

nvidia-smi 

cd /cluster/work/igp_psr/iyakushevsky/GFMdistill

export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

torchrun --nnodes=1 --nproc_per_node=1 main.py +experiment=test_remoteclip_RSITMD_FT_teacher_small

echo "=== Job finished at $(date) ==="