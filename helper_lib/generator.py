"""Image-generation helpers for variational autoencoders and GANs."""

import math
from io import BytesIO

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.utils import make_grid


def generate_samples(model, device="cpu", num_samples=10):
    """Sample a VAE or GAN latent space, plot the images, and return them."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if not hasattr(model, "latent_dim"):
        raise TypeError("model must have a latent_dim attribute")

    device = torch.device(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        latent = torch.randn(num_samples, model.latent_dim, device=device)
        if hasattr(model, "decoder"):
            samples = model.decoder(latent).cpu()
            is_gan = False
        elif hasattr(model, "generator"):
            samples = model.generator(latent).cpu()
            is_gan = True
        elif model.__class__.__name__ in {"Generator", "MNISTGenerator"}:
            samples = model(latent).cpu()
            is_gan = True
        else:
            raise TypeError("model must be a VAE, GAN, or Generator")

    columns = min(6, num_samples)
    rows = math.ceil(num_samples / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows))
    axes = [axes] if num_samples == 1 else axes.flat

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < num_samples:
            image = samples[index]
            if is_gan:
                image = ((image + 1.0) / 2.0).clamp(0.0, 1.0)
            if image.size(0) == 1:
                axis.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            else:
                axis.imshow(image.permute(1, 2, 0).numpy())

    figure.tight_layout()
    plt.show()
    return samples


def generate_sample_tensors(model, device="cpu", num_samples=10):
    """Generate display-ready samples without opening a plot window."""

    if not 1 <= num_samples <= 100:
        raise ValueError("num_samples must be between 1 and 100")

    generator = model.generator if hasattr(model, "generator") else model
    if not hasattr(generator, "latent_dim"):
        raise TypeError("model must be a GAN or generator")

    device = torch.device(device)
    generator.to(device)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, generator.latent_dim, device=device)
        samples = generator(noise)
    return samples.add(1).div(2).clamp(0, 1).cpu()


def samples_to_png(samples):
    """Arrange generated samples in a grid and return PNG bytes."""

    columns = min(10, samples.size(0))
    grid = make_grid(samples, nrow=columns, padding=2)
    image_array = grid.mul(255).byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(image_array)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
