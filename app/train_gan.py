"""Train the Assignment 3 GAN on MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.generator import generate_sample_tensors, samples_to_png
from helper_lib.model import MNISTGAN
from helper_lib.trainer import train_mnist_gan


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gan_mnist_generator.pth"
DEFAULT_PREVIEW_PATH = PROJECT_ROOT / "models" / "gan_mnist_preview.png"


def available_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    device = available_device()
    print(f"Training on {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = MNISTGAN()
    criterion = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(
        model.generator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    train_mnist_gan(
        model,
        data_loader,
        criterion,
        (generator_optimizer, discriminator_optimizer),
        device=device,
        epochs=args.epochs,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.generator.cpu().state_dict(), args.model_path)
    preview = generate_sample_tensors(model.generator, num_samples=64)
    DEFAULT_PREVIEW_PATH.write_bytes(samples_to_png(preview))
    print(f"Saved generator to {args.model_path}")
    print(f"Saved preview to {DEFAULT_PREVIEW_PATH}")


if __name__ == "__main__":
    main()
