"""Evaluation helpers."""

import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    """Return average loss and classification accuracy for ``data_loader``."""
    device = torch.device(device)
    model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    sample_count = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / sample_count if sample_count else 0.0
    accuracy = correct / sample_count if sample_count else 0.0
    print(f"Evaluation - loss: {avg_loss:.4f}, accuracy: {accuracy:.2%}")
    return avg_loss, accuracy
