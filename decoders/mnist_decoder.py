# In a new file, e.g., decoders/mnist_decoder.py

import torch
import torch.nn as nn
from typing import Dict

from decoders.base import Decoder
from encoders.base import Encoder

class ClassificationDecoder(Decoder):
    """
    A self-contained classification model for MNIST that follows the
    framework's expected pattern (takes an encoder in __init__).
    """
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        in_channels: int,
        feature_map_size: int,
        **kwargs
    ):
        super().__init__(encoder=encoder, num_classes=num_classes, finetune=finetune)
        self.model_name = "ClassificationDecoder"
        self.encoder = encoder
        self.finetune = finetune

        # Freeze encoder weights if not finetuning
        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Calculate the input features for the linear layer
        in_features = in_channels * (feature_map_size**2)

        # Define the simple classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=self.num_classes)
        )

    def forward(self, img: Dict[str, torch.Tensor]) -> torch.Tensor:
        # The framework provides a 4D tensor (B, C, T, H, W).
        # Our non-temporal encoder needs a 3D tensor (B, C, H, W).
        # We remove the dummy time dimension (T=1).
        img_no_time = {k: v.squeeze(2) for k, v in img.items()}

        # Get features from the encoder
        if not self.finetune:
            with torch.no_grad():
                features_list = self.encoder(img_no_time)
        else:
            features_list = self.encoder(img_no_time)

        # Select the last feature map from the list and classify
        final_features = features_list[-1]
        logits = self.classifier(final_features)
        
        return logits