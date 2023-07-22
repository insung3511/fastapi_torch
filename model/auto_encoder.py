from typing import Any

import torch.nn.functional as F
import torch.nn as nn
import base64
import torch
import cv2

import numpy as np

class AutoEncoder(nn.Module):
    def __init__(
            self,
            input_shape: tuple = (28, 28),
            output_shape: tuple = (28, 28),
    ):  
        super(AutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.maxpool2d = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.de_conv1 = nn.ConvTranspose2d(256, 128, 3, 1)
        self.de_conv2 = nn.ConvTranspose2d(128, 64, 3, 1)
        self.de_conv3 = nn.ConvTranspose2d(64, 32, 3, 1)
        self.de_conv4 = nn.ConvTranspose2d(32, 1, 3, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))    # Encoder Layer 1
        x = self.maxpool2d(x)           # Output shape: (batch_size, 32, 13, 13)

        x = self.relu(self.conv2(x))    # Encoder Layer 2 
        x = self.maxpool2d(x)           # Output shape: (batch_size, 64, 5, 5)

        x = self.relu(self.conv3(x))    # Encoder Layer 3 
        x = self.maxpool2d(x)           # Output shape: (batch_size, 128, 1, 1)

        x = self.relu(self.conv4(x))    # Latent Layer

        x = self.relu(self.de_conv1(x)) # Decoder Layer 1
        x = self.upsample(x)            # Output shape: (batch_size, 128, 2, 2)

        x = self.relu(self.de_conv2(x)) # Decoder Layer 2
        x = self.upsample(x)            # Output shape: (batch_size, 64, 4, 4)

        x = self.relu(self.de_conv3(x)) # Decoder Layer 3
        x = self.upsample(x)            # Output shape: (batch_size, 32, 8, 8)

        x = self.relu(self.de_conv4(x)) # Decoder Layer 4
        x = self.upsample(x)            # Output shape: (batch_size, 1, 28, 28)

        return x

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