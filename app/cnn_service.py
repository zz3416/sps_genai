"""Load the trained CNN and classify uploaded images."""

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from .cnn_model import CNNClassifier


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_cifar10.pth"

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

_model: CNNClassifier | None = None


def load_model() -> CNNClassifier:
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "The trained CNN model was not found. Run the training script first."
            )

        model = CNNClassifier()
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        _model = model

    return _model


def classify_image(image_bytes: bytes) -> dict[str, object]:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a valid image.") from error

    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    model = load_model()

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        class_index = int(probabilities.argmax().item())

    return {
        "class": CLASSES[class_index],
        "class_index": class_index,
        "confidence": round(float(probabilities[class_index].item()), 4),
    }
