# main.py
from training import train_model
from testing import test_model
from inference import run_inference
import os

if __name__ == "__main__":
    # Check if the best_model.pth file exists
    if os.path.exists('best_model.pth'):
        print("Best model already trained. Skipping training.")
    else:
        # Training
        print("Starting Training")
        print()
        train_model()

    # Testing
    print("Starting Testing")
    print()
    test_model()

    # Inference
    print("Starting Inference")
    print()
    run_inference()