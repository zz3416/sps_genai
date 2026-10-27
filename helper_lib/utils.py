"""Small persistence utilities shared by the notebooks."""

from pathlib import Path

import torch


def save_model(model, file_path):
    """Save a model's learned parameters, creating parent folders as needed."""
    destination = Path(file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    return destination


def load_model(model, file_path, device="cpu"):
    """Load parameters from ``file_path`` into an already-created model."""
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model
