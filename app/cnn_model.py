import torch
from torch import nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """Classify 64x64 RGB images into one of 10 classes."""

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = CNNClassifier()
    sample_images = torch.randn(4, 3, 64, 64)
    output = model(sample_images)

    print(model)
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 10)
