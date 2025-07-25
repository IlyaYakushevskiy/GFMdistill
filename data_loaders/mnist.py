import torch
import torchvision
from torchvision import transforms
from typing import Dict, List
from torchvision.datasets import MNIST as TorchVisionMNIST

class MNIST(torch.utils.data.Dataset):
    """
    A custom Dataset class for MNIST to work with the project's framework.
    It wraps the torchvision.datasets.MNIST class.
    """
    def __init__(
        self,
        root: str,
        split: str,
        bands: Dict[str, List[str]],
        data_mean: Dict[str, List[float]],
        data_std: Dict[str, List[float]],
        num_classes: int,
        classes: List[str],
        ignore_index: int,
        auto_download: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.split = split


        is_train = (split == 'train')
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean['optical'], std=data_std['optical'])
        ])

   
        self.mnist_dataset = TorchVisionMNIST(
            root=root,
            train=is_train,
            download=auto_download,
            transform=transform
        )

        self.num_classes = num_classes
        self.classes = classes
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.mnist_dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dictionary formatted for the project's trainer.
        """
        image, target = self.mnist_dataset[idx]
        
        # Add a dummy time dimension at index 1 -> (C, T, H, W)
        # This changes the shape from [1, 28, 28] to [1, 1, 28, 28]
        image = image.unsqueeze(1)
        
        return {
            "image": {"optical": image},
            "target": torch.tensor(target, dtype=torch.long),
            "metadata": {}
        }