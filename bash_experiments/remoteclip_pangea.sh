export CUDA_VISIBLE_DEVICES=0
export RANK=0

torchrun run_pangaea.py \
   --config-name=train \
   dataset=fivebillionpixels \
   encoder=remoteclip \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation