# config.py
class Config:
    TRAIN_DIR = "./cars_data/train"
    VAL_DIR = "./cars_data/val"
    TEST_DIR = "./cars_data/test"
    INFERENCE_DIR = "./cars_data/inference"
    BATCH_SIZE = 8
    LR = 0.00001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    NUM_CLASSES = 4
