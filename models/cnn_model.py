import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network for retinal vessel segmentation.
    Extracts spatial features using sequential convolutions and max pooling.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Feature Extraction Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 1x1 Convolution for final pixel-wise classification
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        
        # Downsampling
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Output Block (Probabilities between 0 and 1)
        x = self.conv3(x)
        x = self.sigmoid(x)
        
        return x