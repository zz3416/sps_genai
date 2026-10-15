"""Train the Assignment 2 CNN on CIFAR-10 and save its weights."""

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .cnn_model import CNNClassifier


BATCH_SIZE = 32
EPOCHS = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "cnn_cifar10.pth"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_data_loaders() -> tuple[DataLoader, DataLoader]:
    # CIFAR-10 images are 32x32, while the assignment CNN expects 64x64.
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader


def train_model(
    model: CNNClassifier,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = outputs.argmax(dim=1)
            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

        average_loss = running_loss / len(train_loader)
        accuracy = 100 * running_correct / running_total
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"loss: {average_loss:.4f} - accuracy: {accuracy:.2f}%"
        )


def evaluate_model(
    model: CNNClassifier,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    return accuracy


def main() -> None:
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = create_data_loaders()
    model = CNNClassifier().to(device)

    train_model(model, train_loader, device)
    evaluate_model(model, test_loader, device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
