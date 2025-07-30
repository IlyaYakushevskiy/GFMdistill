import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List
from logging import Logger
from encoders.base import Encoder



class SimpleCNNEncoder(Encoder):
    """
    A simple CNN encoder for MNIST, designed as a 'Teacher' model for distillation experiments.
    """

    def __init__(
        self,
        input_bands: Dict[str, List[str]],
        output_layers: List[int] = [1, 2],
        num_blocks : int = 2,
        base_channels: int = 32,
        #save_feature_maps: bool = False,
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
        # C * 2 ^ num blocks 
        output_dims_map = {i: base_channels * (2 ** (i - 1)) for i in range(1, num_blocks + 1)}
        selected_output_dims = [output_dims_map[i] for i in output_layers]

        super().__init__(
            model_name="simple_cnn_encoder",
            encoder_weights=None,  # This model is trained from scratch
            input_bands=input_bands,
            input_size=28,  # MNIST specific
            embed_dim=selected_output_dims[-1],  # The embedding dimension of the final layer
            output_layers=output_layers,
            output_dim=selected_output_dims,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=True,  # return features from multiple scales
            download_url=None
        )

        # ---- Define Network Architecture ----

        self.blocks = nn.ModuleList()
        #self.save_feature_maps = save_feature_maps
        in_channels = 1

        for i in range(1, num_blocks + 1):
            out_channels = base_channels * (2 ** (i - 1))
            block = nn.Sequential(
                OrderedDict(
                    [
                        (f"conv{i}", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
                        (f"relu{i}", nn.ReLU(inplace=True)),
                        (f"pool{i}", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ]
                )
            )
            self.blocks.append(block)
            in_channels = out_channels
        
        #can be used to reduce the feature maps to a single vector
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP reduces [B, C, H, W] -> [B, C, 1, 1]

        #WE SHALL PRODUCE THE FEATURE MAPS ONLY! 
        
        #self.fc = nn.Linear(output_dims_map[num_blocks], num_classes)  # Final logits layer

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
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            if i in self.output_layers:
                outputs.append(x)

        # if self.save_feature_maps == True:
        #     # Apply Global Average Pooling to the last feature map
        #     print("SAVING THE MAPS")
            # x = self.gap(x)
            # outputs.append(x)
        #print(f"Feature maps from layers {self.output_layers}: {[o.shape for o in outputs]}")
        return outputs #shape (1,512 for 4 layers )
    
     