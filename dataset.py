# dataset.py
import os
import pathlib
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    # Find and sort the classes found in the dataset directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    # Map class names to indices
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

class CarsDataset(Dataset):
    def __init__(self, target_directory: str, transform=None):
        # List all image paths in the dataset directory
        self.paths = list(pathlib.Path(target_directory).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_directory)

    def __load_image__(self, index: int) -> Image.Image:
        # Load an image from the specified index
        return Image.open(self.paths[index])

    def __len__(self) -> int:
        # Return the total number of images in the dataset
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Get an image and its corresponding class index
        image = self.__load_image__(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(image), class_idx
        else:
            return image, class_idx 