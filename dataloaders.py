# dataloaders.py
import os
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CarsDataset
from config import Config

class_names_found = sorted([entry.name for entry in list(os.scandir(Config.TRAIN_DIR))])

# Define image transformations for training and validation sets
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create instances of the custom dataset for training and validation
train_dataset = CarsDataset(Config.TRAIN_DIR, transform=train_transform)
val_dataset = CarsDataset(Config.VAL_DIR, transform=val_transform)

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)