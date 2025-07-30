import torch
import torch.nn as nn
from typing import Dict

from decoders.base import Decoder
from encoders.base import Encoder

class ClassificationDecoder(Decoder):
    """
    A self-contained classification model that automatically adapts to the encoder's output.
    """
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        # REMOVED: in_channels and feature_map_size are no longer needed here
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
        
        # === Step 1: Dynamically determine the encoder's output feature size ===
        # Create a dummy input tensor matching the encoder's expected input size
        dummy_input = torch.randn(1, 1, self.encoder.input_size, self.encoder.input_size)
        
        # Perform a dummy forward pass through the encoder to get the output shape
        with torch.no_grad():
            dummy_features_list = self.encoder({'optical': dummy_input})
        
        # Get the final feature map and calculate its flattened size
        final_feature_map = dummy_features_list[-1]
        in_features = final_feature_map.view(1, -1).size(1) # e.g., converts [1, 512, 1, 1] to 512

        print(f"Decoder dynamically initialized with in_features = {in_features}")
        
        # === Step 2: Define the classifier head with the correct input size ===
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=self.num_classes)
        )

    def forward(self, img: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Remove the dummy time dimension (T=1)
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
        
        return logits, features_list