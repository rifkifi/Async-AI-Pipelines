import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Device setup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Running on:", device)


# MNIST preprocessing
# Resize images to 14x14


transform = transforms.Compose([
    transforms.Resize((14, 14)),
    transforms.ToTensor()
])


# Dataset loading


train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)


# CNN Model


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        # 14x14x1 -> 12x12x8
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3
        )

        self.relu = nn.ReLU()

        # 12x12 -> 6x6
        self.pool = nn.MaxPool2d(2)

        # Flatten = 6*6*8 = 288
        self.fc1 = nn.Linear(288, 16)

        # Output layer
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):

        # Convolution
        x = self.conv1(x)

        # Activation
        x = self.relu(x)

        # Max Pooling
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layer
        x = self.fc1(x)

        x = self.relu(x)

        # Final output
        x = self.fc2(x)

        return x


# Model initialization


model = CNN().to(device)

print(model)


# Loss and optimizer


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training


epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print("Epoch:", epoch + 1, "Loss:", round(avg_loss, 4))


# Testing


model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("\nTest Accuracy:", round(accuracy, 2), "%")


# Save model


torch.save(model.state_dict(), "mnist_cnn.pth")

print("\nModel saved successfully")
