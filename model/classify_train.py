from deep_learning_model import MNIST_Classify_Model
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

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

model = MNIST_Classify_Model().to(device)
critention = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH):
    print("- " * 20)

    model.train()
    train_loss, valid_loss = 0.0, 0.0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        optimizer.zero_grad()
        loss = critention(output, labels)
        loss.backward()

        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(train_dataloader.dataset)
    print(f"[TRAIN] Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = critention(output, labels)
            valid_loss += loss.item() * images.size(0)
        valid_loss = valid_loss / len(test_dataloader.dataset)
        print(f"[VALID] Epoch: {epoch + 1}, Validation Loss: {valid_loss:.4f}")

print("Finished Training\n[INFO] Saving Model...")
torch.save(model.state_dict(), 'model.pth')
print("Finished Saving Model\nExiting...")

