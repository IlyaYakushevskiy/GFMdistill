import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List
from logging import Logger
from encoders.base import Encoder


class SimpleCNNEncoder(Encoder):
    """
    A simple CNN encoder for MNIST, designed as a 'Teacher' model for distillation experiments.
    This architecture follows the provided template and the previously discussed model design.
    """

    def __init__(
        self,
        input_bands: Dict[str, List[str]],
        output_layers: List[int] = [1, 2],
        **kwargs
    ):
        """
        Initializes the SimpleCNNEncoder.
        Args:
            input_bands (Dict[str, List[str]]): Dictionary specifying the input bands. For MNIST,
                                                 this would be e.g., {'optical': ['grayscale']}.
            output_layers (List[int]): A list containing the block numbers from which to
                                       extract feature maps. Defaults to [1, 2].
        """
        # The channel dimensions of the output of each block
        output_dims_map = {1: 32, 2: 64}
        selected_output_dims = [output_dims_map[i] for i in output_layers]

        super().__init__(
            model_name="simple_cnn_encoder",
            encoder_weights=None,  # This model is trained from scratch
            input_bands=input_bands,
            input_size=28,  # MNIST specific
            embed_dim=64,  # The embedding dimension of the final layer
            output_layers=output_layers,
            output_dim=selected_output_dims,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=True,  # We return features from multiple scales
            download_url=None,
        )

        # ---- Define Network Architecture ----

        # Convolutional Block 1
        self.block1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(1, 32, kernel_size=3, padding=1)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("pool", nn.MaxPool2d(kernel_size=2, stride=2)),
                ]
            )
        )
        # Output shape: [B, 32, 14, 14]

        # Convolutional Block 2
        self.block2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(32, 64, kernel_size=3, padding=1)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("pool", nn.MaxPool2d(kernel_size=2, stride=2)),
                ]
            )
        )
        # Output shape: [B, 64, 7, 7]

    def load_encoder_weights(self, logger: Logger) -> None:
        """
        This model is designed to be trained from scratch on MNIST.
        Therefore, this method does nothing.
        """
        logger.info(f"'{self.model_name}' is initialized with random weights (training from scratch).")
        return
    
    def forward(self, image: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Performs a forward pass through the encoder.
        Args:
            image (Dict[str, torch.Tensor]): A dictionary containing the input tensor.
                                              The key should match the one in `input_bands`.
                                              e.g., {'optical': <tensor of shape [B, 1, 28, 28]>}
        Returns:
            List[torch.Tensor]: A list of feature maps from the specified output_layers.
        """
        # Assuming the input tensor is accessible via the 'optical' key
        x = image["optical"]

        outputs = []

        # Pass input sequentially through the blocks
        x_block1 = self.block1(x)
        if 1 in self.output_layers:
            outputs.append(x_block1)

        x_block2 = self.block2(x_block1)
        if 2 in self.output_layers:
            outputs.append(x_block2)

        return outputs