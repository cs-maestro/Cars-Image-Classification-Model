# Cars Image Classification Model

This repository contains code for a car classification project using PyTorch. The project involves training a model to classify car images into different categories such as convertible, sedan, SUV, and truck.

## Project Structure

The project is organized into several files and directories:

- **cars_data:** This directory contains directories for inference, test, train, and val with their respective image files.

- **main.py:** The main script to execute training, testing, and inference.

- **constants.py:** Contains constants used throughout the project.

- **model.py:** Defines the model for car classification.

- **dataset.py:** Defines the custom dataset and helper functions to load and preprocess data.

- **dataloaders.py:** Creates data loaders for training and validation sets.

- **training.py:** Contains the training loop and related functions.

- **testing.py:** Evaluate the model on a separate test set.

- **inference.py:** Runs inference on custom images (you can add more).

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cs-maestro/Cars-Image-Classification-Model.git
   cd Cars-Image-Classification-Model

2. **Install dependencies:**

   Install the missing (if any) dependencies for the project.
   
3. **Run main.py:**

   Run the main file to generate the model and then test it on val and inference data.
   ```bash
   python main.py

## Model Architecture
The model architecture is based on VGG19 with modifications. Batch normalization is applied after convolutional layers, and the classifier is customized for the car classification task.

## Contributions
Contributions are welcome! If you find any issues or improvements, please open an issue or submit a pull request.
