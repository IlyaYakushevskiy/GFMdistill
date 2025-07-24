import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import open_clip
from sklearn.metrics import accuracy_score
from glob import glob
import yaml


#custom imports
# import encoders.remoteclip_encoder
# import decoders.upernet_decoder
# import data_loaders.fbp
import encoders.pos_embed

from encoders.remoteclip_encoder import RemoteCLIP_Encoder
from decoders.upernet import SegUPerNet
from  data_loaders.fbp import FiveBillionPixels

from tqdm import tqdm
import os
import argparse
from torchmetrics import JaccardIndex
import open_clip  # Used only for the image transform
import logging # Required for the encoder's weight loading method

from huggingface_hub import hf_hub_download
import torch, open_clip
from PIL import Image
import  numpy as np


#GLOBAL VARIABLES
# images_path = "./data/FBP/Image__8bit_NirRGB/"
# labels_path = "./data/FBP/Annotation__index/"


#download RemoteCLIP from HF 

model_name = 'ViT-B-32' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

path_to_your_checkpoints = './checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'

ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location="cpu")
message = model.load_state_dict(ckpt)
print(message)


config_path = "./config/remoteclip.yaml"
ds_config_path = "./config/fivebillionpixels.yaml"
weights_path = "./checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
num_classes = 25 
batch_size = 2 #could only run with batch 2 on mac cpu
workers = 0 #mb problems on cpu 

#split the data! 

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    encoder_params = {
        'encoder_weights': weights_path,
        'input_bands': {'optical': ['B8','B4', 'B3', 'B2']},
        'input_size': 224,
        'embed_dim': 768,
        'patch_size': 32,
        'width': 768,
        'head_width': 64,
        'layers': 12,
        'mlp_ratio': 4.0,
        'output_layers': [3, 5, 7, 11], # Extract features from these 4 blocks
        'output_dim': 768,
        'download_url': None, 
    }
    

    encoder = RemoteCLIP_Encoder(**encoder_params)
    
    
    print("Loading RemoteCLIP encoder weights...")
    
    logging.basicConfig()
    logger = logging.getLogger("MyLogger")
    encoder.load_encoder_weights(logger=logger)
    print("Encoder weights loaded successfully.")


    model = SegUPerNet(
        encoder=encoder,
        num_classes=num_classes,
        finetune=False,  # Set to False to freeze the encoder during inference
        channels=512     # Internal channels for the decoder
    ).to(device)
    model.eval()


    #_, _, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

    #dataset config 


    with open(ds_config_path, 'r') as f:
        dataset_params = yaml.safe_load(f)

    # --- CREATE YOUR CUSTOM 4-CHANNEL TRANSFORM (by LLM), remoteclip is pretrained on RGB; wether I should include 4th 
    # channel is dillema cause of pretrain-FT data mismatch, but the information in NIR might be crucial for the task

    # Get the mean and std for your 4 bands from the config
    mean_4_channels = dataset_params['data_mean']['optical']
    std_4_channels = dataset_params['data_std']['optical']
  
    custom_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_4_channels, std=std_4_channels)
    ])  
    dataset_params.pop('_target_')
    print(f"Dataset parameters loaded: {dataset_params}")

    dataset = FiveBillionPixels(
        split='test',
        transform=custom_transform,
        **dataset_params
    )
    dataloader = DataLoader(
        dataset,
        batch_size= batch_size,
        num_workers= workers,
        shuffle=False
    ) #returns output with 3 keys: image, target, metadata
    print(f"Dataset loaded with {len(dataset)} samples.")

    #  mIoU
    jaccard = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)
    
    print("\nStarting inference and mIoU calculation...")

    for batch in tqdm(dataloader, desc="Evaluating"):
        img_dict = {
            'optical': batch['image']['optical'].to(device)
        }
        masks = batch['target'].to(device)
        output_shape = masks.shape[-2:]

        with torch.no_grad():
            logits = model(img_dict, output_shape=output_shape)
            preds = logits.argmax(dim=1)

        jaccard.update(preds, masks)

    # --- 6. Compute and Print Final mIoU ---
    miou_score = jaccard.compute()
    print(f"\n--- Evaluation Finished ---")
    print(f"Mean Intersection over Union (mIoU): {miou_score.item():.4f}")
    print(f"---------------------------\n")


if __name__ == "__main__":
    main()

