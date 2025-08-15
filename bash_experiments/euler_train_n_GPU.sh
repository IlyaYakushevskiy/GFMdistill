#!/bin/bash
#SBATCH -A es_schin
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gpus-per-node=2
# #SBATCH --gpus=rtx_4090:2
#SBATCH --gres=gpumem:22G
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time 1:00:00
#SBATCH -o job_output/train_rc_RSITMD_2GPU%j.out
#SBATCH -e job_output/train_rc_RSITMD__2GPU%j.err

echo "=== Job starting on $(hostname) at $(date) ==="

module eth_proxy load stack/2024-05 gcc/13.2.0 cuda/12.1.1 python/3.11.6_cuda

source ~/euler_env/bin/activate
echo "Activated Python venv: $(which python)"

nvidia-smi 

python - <<'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA version in Torch:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name())
EOF

cd /cluster/work/igp_psr/iyakushevsky/GFMdistill

torchrun --nproc_per_node=2 main.py +experiment=train_remoteclip_RSITMD_small

echo "=== Job finished at $(date) ==="
