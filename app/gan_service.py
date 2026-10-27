"""Load the trained MNIST generator and return generated image grids."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from helper_lib.generator import generate_sample_tensors, samples_to_png
from helper_lib.model import MNISTGenerator


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gan_mnist_generator.pth"


@lru_cache(maxsize=1)
def load_generator() -> MNISTGenerator:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained GAN generator was not found. "
            "Run `uv run python -m app.train_gan` first."
        )

    generator = MNISTGenerator()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def generate_image_grid(num_samples: int) -> bytes:
    generator = load_generator()
    samples = generate_sample_tensors(generator, num_samples=num_samples)
    return samples_to_png(samples)
