#!/bin/bash
#SBATCH -A es_schin
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:22G
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time 3:00:00
#SBATCH -o job_output/train_rc_RSITMD_distill%j.out
#SBATCH -e job_output/train_rc_RSITMD__distill%j.err

echo "=== Job starting on $(hostname) at $(date) ==="
echo "=== SLURM_PROCID: $SLURM_PROCID, SLURM_LOCALID: $SLURM_LOCALID, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES ==="

module eth_proxy load stack/2024-05 gcc/13.2.0 cuda/12.1.1 python/3.11.6_cuda

source ~/euler_env/bin/activate
echo "Activated Python venv: $(which python)"

# The nvidia-smi command will now run for each task, showing the single GPU it has been allocated
nvidia-smi 

# We no longer need the python check block, but it's harmless to keep.

cd /cluster/work/igp_psr/iyakushevsky/GFMdistill

# --- CORRECTED: Use srun to launch python directly. No more torchrun. ---
# PyTorch will automatically detect the Slurm environment variables to set up distributed training.

torchrun --nproc_per_node=2 main.py +experiment=train_remoteclip_RSITMD_small_distill

echo "=== Job finished at $(date) ==="