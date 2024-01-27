# training.py
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import sys

from model import get_model
from dataloaders import train_loader, val_loader
from config import Config

# Define global variable(s)
global best_val_loss

# Setup device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# Set random seed for reproducibility
torch.manual_seed(30)

# Instantiate the model, loss function, optimizer, and scheduler
model = get_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Training loop with learning rate scheduler
best_val_loss = float('inf')

# Training loop
def train_model():
    global best_val_loss  # Declare it as a global variable
    for epoch in range(Config.EPOCHS):
        model.train()
        for images, labels in train_loader:
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)
            # Zero the gradients, forward pass, backward pass, and optimizer step
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                # Move data to GPU
                images, labels = images.to(device), labels.to(device)
                # Forward pass and calculate loss
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Update learning rate
        scheduler.step()

        # Save model with best validation performance
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # Print epoch statistics
        logging.info(f'Epoch {epoch + 1}/{Config.EPOCHS}, Loss: {loss.item():.4f}, Validation Accuracy: {100 * correct / total:.2f}%')

# Train the model
if __name__ == "__main__":
    train_model()