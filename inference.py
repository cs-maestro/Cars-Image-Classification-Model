# inference.py
from PIL import Image
import torch
import logging
import sys
import os

from dataloaders import val_transform, class_names_found
from model import get_model
from config import Config

# Setup device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# Run the model on custom images
def run_inference():
    # Get a list of all files in the inference directory
    all_files = os.listdir(Config.INFERENCE_DIR)

    # filter the list for files ending in .jpg
    image_files = [file for file in all_files if file.endswith('.jpg')]

    # now you can loop over the images
    for i in range(len(image_files)):
        image = Image.open(os.path.join(Config.INFERENCE_DIR, image_files[i]))
        model = get_model().to(device)
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        image = val_transform(image)
        image = image.unsqueeze(0).to(device)
        outputs = model(image)


        # Print the predicted class with confidence score
        _, predicted = torch.max(outputs.data, 1)
        class_idx = predicted.cpu().numpy()[0]
        print(f"Image: {image_files[i]}")
        print(f"Predicted class: {class_names_found[class_idx]}")
        print()

# Run inference
if __name__ == "__main__":
    run_inference()