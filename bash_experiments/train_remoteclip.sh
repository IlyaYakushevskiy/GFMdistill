RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python run.py \
  --config-name=train \
  dataset=spacenet7 \
  encoder=remoteclip \
  decoder=seg_upernet_mt_ltae \
  preprocessing=seg_default \
  criterion=cross_entropy \
  task=segmentation
