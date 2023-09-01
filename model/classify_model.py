from typing import Any

import torch.nn.functional as F
import torch.nn as nn
import logging
import base64
import torch
import cv2

import numpy as np

logging.basicConfig(level=logging.INFO)

class MNIST_Classify_Model(nn.Module):
    def __init__(self, input_shape: tuple = (28, 28), output_shape: tuple = (10)):
        super(MNIST_Classify_Model, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.maxpool2d = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Feature Extraction Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
       
        # Feature Extraction Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        
        # Feature Extraction Layer 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool2d(x)

        # Classification Layer
        x = self.flatten(x)                 # Shape: (batch_size, 256, 1, 1) -> (batch_size, 256)
        x = F.relu(self.fc1(x))             # Shape: (batch_size, 256) -> (batch_size, 64)
        x = F.softmax(self.fc2(x), dim=1)   # Shape: (batch_size, 64) -> (batch_size, 10)

        return x

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
class DataPreprocessing:
    def __init__(
            self, 
            target_datatype: np.float32 = None, 
            image_width: int = 28,
            image_height: int = 28,
            image_channel: int = 1
        ):
        self.target_datatype = target_datatype
        if self.target_datatype is None: ValueError(f"target_datatype must be specified. (e.g. np.float32)\nExcepted: {np.float32}, Input: {self.target_datatype}")

        self.image_width = image_width
        self.image_height = image_height
        self.image_channel = image_channel
        if self.image_width is not int or self.image_height is not int or self.image_channel is not int: 
            ValueError(f"image_width, image_height, image_channel must be specified. (e.g. 28, 28, 1)\nExcepted: {int}, Input: {self.image_width, self.image_height, self.image_channel}")

    def __call__(self, image: np.ndarray) -> torch.tensor:
        image = base64.b64decode(image)
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.resize(image, (28, 28))
        image = image.reshape(-1, self.image_channel, self.image_width, self.image_height)
        image = image / 255.0
        image = image.astype(self.target_datatype)

        return image