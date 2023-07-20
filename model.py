from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch

device = torch.device('mps')

train_datasets = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(train_datasets, batch_size=32, shuffle=True)

test_datasets = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(test_datasets, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
    nn.Softmax(dim=1)
).to(device)

critention = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        optimizer.zero_grad()
        loss = critention(output, labels)
        loss.backward()

        optimizer.step()

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = critention(output, labels)

            print(loss.item())

torch.save(model.state_dict(), 'model.pth')

