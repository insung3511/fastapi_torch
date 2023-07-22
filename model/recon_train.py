from torch.utils.data import DataLoader
from auto_encoder import AutoEncoder
from torchvision import transforms
from torchvision import datasets
from torchsummary import summary

import torch.optim as optim
import torch.nn as nn
import torch

device = torch.device('mps')
EPOCH = 100
BATCH_SIZE = 32
LEARNING_RATE = 3e-3

train_datasets = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)

test_datasets = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=True)

model = AutoEncoder().to(device)
critention = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(summary(model.to(device='cpu'), (1, 28, 28), device='cpu', batch_size=BATCH_SIZE))