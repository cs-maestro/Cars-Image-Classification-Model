# model.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG19_Weights

from config import Config

# Define the model using VGG19 as a backbone
class FeatureExtractorModel(nn.Module):
    def __init__(self):
        super(FeatureExtractorModel, self).__init__()
        self.backbone = models.vgg19(weights=VGG19_Weights.DEFAULT)
        # Remove the final classification layer
        self.backbone.classifier = nn.Identity()
        # Batch normalization after convolutional layers
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.batch_norm2 = nn.BatchNorm2d(512)

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, Config.NUM_CLASSES),
        )


    def forward(self, x):
        x = self.backbone(x)  # Extract features from VGG19
        x = x.view(x.size(0), -1)  # Flatten features
        x = self.classifier(x)  # Classify
        return x
    
# Instantiate the model
def get_model():
    return FeatureExtractorModel()