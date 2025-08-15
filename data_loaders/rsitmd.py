import os
import json
import numpy as np
import tifffile as tiff
import torch
from data_loaders.base import RawGeoFMDataset
from typing import Callable, Optional
from pathlib import Path

class RSITMD(RawGeoFMDataset):
    def __init__(
        self,
        use_cmyk: bool,
        split: str,
        classes: list,
        root_path: str,
        dataset_name: str,
        num_classes: int,
        transform: Optional[Callable] = None,
        json_filename: str = "dataset_RSITMD_split.json", 
        **kwargs,
    ):
        """
        DataLoader for the RSITMD dataset, adapted for image classification.
        """
        # Pass only relevant arguments to the parent class
        super().__init__(
            split=split,
            root_path=root_path,
            classes=classes,
            dataset_name=dataset_name,
            num_classes=num_classes,
            **kwargs
        )
        
        self.transform = transform
        self.use_cmyk = use_cmyk
        self.json_filename = json_filename # Store the name of the JSON file
        # Create a mapping from class name string to integer index
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # This list will store tuples of (image_path, class_index)
        self.samples = []
        
        # Load and parse the JSON file to build the dataset
        self._load_samples()

    def _load_samples(self):
        """Parses the JSON file to find all images and labels for the current split."""
        # Use Path for robust path handling
        root_path = Path(self.root_path)
        json_path = root_path / self.json_filename
        images_dir = root_path / "images"
        
        print(f"INFO: Loading RSITMD '{self.split}' split from {json_path}")

        if not json_path.is_file():
            raise FileNotFoundError(f"JSON file not found at {json_path}. Did you run the split creation script?")

        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            
        for image_info in data["images"]:
            # This is the core logic. Check if the entry's split matches the one we want.
            if image_info["split"] == self.split:
                filename = image_info["filename"]
                class_name = filename.split('_')[0]
                
                if class_name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_name]
                    img_path = images_dir / filename
                    
                    # Double-check that the image file actually exists
                    if img_path.is_file():
                        self.samples.append((img_path, class_idx))
                    else:
                        print(f"Warning: JSON lists file '{filename}' for split '{self.split}', but file not found at {img_path}")

        if not self.samples:
            raise RuntimeError(f"Found 0 images in split '{self.split}' at path {self.root_path} using JSON file {self.json_filename}. Please check your JSON file.")
            
        print(f"INFO: Loaded {len(self.samples)} samples for split '{self.split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        img_path, class_idx = self.samples[index]
        image_array = tiff.imread(img_path)
        image_tensor = torch.from_numpy(image_array.astype(np.float32)).permute(2, 0, 1)
        target = torch.tensor(class_idx, dtype=torch.long)
        
        data = {
            "image": { "optical": image_tensor.unsqueeze(1) },
            "target": target,
            "metadata": {"filepath": str(img_path)},
        }
        
        if self.transform:
            data = self.transform(data)
            
        return data