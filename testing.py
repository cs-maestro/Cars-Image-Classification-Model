# testing.py
import torch
from torch.utils.data import DataLoader
import logging
import sys

from dataset import CarsDataset
from dataloaders import val_transform
from model import get_model
from config import Config

# Setup device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# Test dataset and loader
test_dataset = CarsDataset(Config.TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Test the model on the test set
def test_model():
    model = get_model().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)
            # Forward pass and calculate accuracy
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Print test accuracy
    logging.info(f'Test Accuracy: {100 * test_correct / test_total:.2f}%')

# Test the model
if __name__ == "__main__":
    test_model()