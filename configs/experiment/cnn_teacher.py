import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict
from typing import Dict


class SimpleCNNDecoder(nn.Module):
    """
    A simple decoder for MNIST classification. It takes the final feature map
    from an encoder, flattens it, and passes it through a linear layer to
    produce class logits.
    """

    def __init__(self, in_channels: int, feature_map_size: int, num_classes: int, **kwargs):
        """
        Initializes the SimpleCNNDecoder.
        Args:
            in_channels (int): Number of channels in the input feature map.
                               For the teacher encoder, this would be 64.
            feature_map_size (int): The spatial size (height or width) of the input
                                    feature map. For the teacher encoder, this is 7.
            num_classes (int): The number of output classes (e.g., 10 for MNIST).
        """
        super().__init__()
        self.model_name = "simple_cnn_decoder"
        
        # Calculate the total number of features after flattening
        in_features = in_channels * (feature_map_size**2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    def forward(self, image: Dict[str, torch.Tensor]) -> torch.Tensor:
      
            encoder_output_list = self.encoder(image)
            
            # final feature map from that list
            final_features = encoder_output_list[-1]
            
            # We pass ONLY that single feature tensor to the decoder head
            logits = self.decoder_head(final_features)
            
            return logits