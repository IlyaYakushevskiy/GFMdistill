python run_remoteclip.py
Using device: cpu
Loading RemoteCLIP encoder weights...
WARNING:MyLogger:Incompatible parameters:
conv1.weight: expected torch.Size([768, 4, 32, 32]) but found torch.Size([768, 3, 32, 32])
Encoder weights loaded successfully.
Dataset parameters loaded: {'dataset_name': 'FiveBillionPixels', 'root_path': './data/FBP/', 'download_url': False, 'auto_download': False, 'use_cmyk': False, 'img_size': 520, 'multi_temporal': False, 'multi_modal': False, 'ignore_index': 0, 'num_classes': 25, 'classes': ['unlabeled', 'industrial area', 'paddy field', 'irrigated field', 'dry cropland', 'garden land', 'arbor forest', 'shrub forest', 'park', 'natural meadow', 'artificial meadow', 'river', 'urban residential', 'lake', 'pond', 'fish pond', 'snow', 'bareland', 'rural residential', 'stadium', 'square', 'road', 'overpass', 'railway station', 'airport'], 'distribution': [0.0, 0.0368, 0.0253, 0.3567, 0.0752, 0.0095, 0.0694, 0.0096, 0.0004, 0.0055, 0.0025, 0.0568, 0.0548, 0.1396, 0.0102, 0.0129, 0.0004, 0.0456, 0.0447, 0.0003, 0.0002, 0.0383, 0.0025, 0.0007, 0.0011], 'bands': {'optical': ['B8', 'B4', 'B3', 'B2']}, 'data_mean': {'optical': [92.6, 124.3, 94.2, 98.0]}, 'data_std': {'optical': [44.5, 51.0, 50.0, 47.1]}, 'data_min': {'optical': [0.0, 0.0, 0.0, 0.0]}, 'data_max': {'optical': [0.0, 0.0, 0.0, 0.0]}}
Loading images from: ./data/FBP/Image__8bit_NirRGB/*.tif
Dataset loaded with 150 samples.

Starting inference and mIoU calculation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [51:00<00:00, 40.81s/it]

--- Evaluation Finished ---
Mean Intersection over Union (mIoU): 0.0023