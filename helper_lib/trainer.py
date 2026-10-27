"""Training helpers."""

import torch
import torch.nn.functional as F
from torch import optim


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Train ``model`` for ``epochs`` full passes over ``data_loader``."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    device = torch.device(device)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        average_loss = running_loss / sample_count if sample_count else 0.0
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f}")

    return model


def vae_loss(reconstruction, inputs, mu, logvar, beta=500.0):
    """Return the Module 5 reconstruction plus KL-divergence VAE loss."""
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction, inputs, reduction="sum"
    )
    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return beta * reconstruction_loss + kl_divergence


def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
):
    """Train a VAE and return it after ``epochs`` full data passes."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    device = torch.device(device)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            reconstruction, mu, logvar = model(inputs)
            loss = criterion(reconstruction, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item()
            sample_count += batch_size

        average_loss = running_loss / sample_count if sample_count else 0.0
        print(f"Epoch {epoch + 1}/{epochs} - VAE loss: {average_loss:.4f}")

    return model


def _gan_optimizers(model, optimizer, learning_rate):
    """Normalize supported optimizer inputs into generator/critic optimizers."""
    if optimizer is None:
        return (
            optim.RMSprop(model.generator.parameters(), lr=learning_rate),
            optim.RMSprop(model.critic.parameters(), lr=learning_rate),
        )
    if isinstance(optimizer, dict):
        try:
            return optimizer["generator"], optimizer["critic"]
        except KeyError as error:
            raise ValueError(
                "optimizer dict must contain 'generator' and 'critic'"
            ) from error
    if isinstance(optimizer, (tuple, list)) and len(optimizer) == 2:
        return optimizer[0], optimizer[1]
    raise TypeError(
        "optimizer must be None, a (generator, critic) pair, or a dict with "
        "'generator' and 'critic' entries"
    )


def train_gan(
    model,
    data_loader,
    criterion=None,
    optimizer=None,
    device="cpu",
    epochs=10,
    n_critic=5,
    clip_value=0.01,
    learning_rate=5e-5,
):
    """Train the Module 6 Wasserstein GAN and return the GAN container.

    ``criterion`` is accepted for compatibility with the activity's requested
    signature. WGAN uses mean critic scores directly, so no criterion object is
    needed. ``optimizer`` may be ``None``, a two-item ``(generator, critic)``
    sequence, or a dictionary containing those two keys.
    """
    del criterion
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if n_critic < 1:
        raise ValueError("n_critic must be at least 1")
    if not hasattr(model, "generator") or not hasattr(model, "critic"):
        raise TypeError("model must be the GAN returned by get_model('GAN')")

    device = torch.device(device)
    model.to(device)
    generator_optimizer, critic_optimizer = _gan_optimizers(
        model, optimizer, learning_rate
    )
    history = []

    for epoch in range(epochs):
        model.generator.train()
        model.critic.train()
        generator_loss = 0.0
        critic_loss = 0.0
        batch_count = 0

        for batch in data_loader:
            real_images = batch[0] if isinstance(batch, (tuple, list)) else batch
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            for _ in range(n_critic):
                noise = torch.randn(
                    batch_size, model.latent_dim, 1, 1, device=device
                )
                fake_images = model.generator(noise).detach()
                loss_critic = -(
                    model.critic(real_images).mean()
                    - model.critic(fake_images).mean()
                )

                critic_optimizer.zero_grad()
                loss_critic.backward()
                critic_optimizer.step()
                with torch.no_grad():
                    for parameter in model.critic.parameters():
                        parameter.clamp_(-clip_value, clip_value)

            noise = torch.randn(
                batch_size, model.latent_dim, 1, 1, device=device
            )
            fake_images = model.generator(noise)
            loss_generator = -model.critic(fake_images).mean()

            generator_optimizer.zero_grad()
            loss_generator.backward()
            generator_optimizer.step()

            critic_loss += loss_critic.item()
            generator_loss += loss_generator.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("data_loader did not yield any batches")
        epoch_metrics = {
            "epoch": epoch + 1,
            "critic_loss": critic_loss / batch_count,
            "generator_loss": generator_loss / batch_count,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"critic loss: {epoch_metrics['critic_loss']:.4f}, "
            f"generator loss: {epoch_metrics['generator_loss']:.4f}"
        )

    model.training_history = history
    return model


def train_mnist_gan(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
):
    """Train the Assignment 3 MNIST GAN with binary cross-entropy loss."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not hasattr(model, "generator") or not hasattr(model, "discriminator"):
        raise TypeError(
            "model must be the MNISTGAN returned by get_model('MNISTGAN')"
        )
    if not isinstance(optimizer, (tuple, list)) or len(optimizer) != 2:
        raise TypeError(
            "optimizer must be a (generator_optimizer, discriminator_optimizer) pair"
        )

    generator_optimizer, discriminator_optimizer = optimizer
    device = torch.device(device)
    model.to(device)
    history = []

    for epoch in range(epochs):
        model.generator.train()
        model.discriminator.train()
        generator_loss_total = 0.0
        discriminator_loss_total = 0.0
        batch_count = 0

        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            discriminator_optimizer.zero_grad()
            real_predictions = model.discriminator(real_images)
            real_loss = criterion(real_predictions, real_targets)

            noise = torch.randn(batch_size, model.latent_dim, device=device)
            fake_images = model.generator(noise)
            fake_predictions = model.discriminator(fake_images.detach())
            fake_loss = criterion(fake_predictions, fake_targets)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            noise = torch.randn(batch_size, model.latent_dim, device=device)
            fake_images = model.generator(noise)
            predictions = model.discriminator(fake_images)
            generator_loss = criterion(predictions, real_targets)
            generator_loss.backward()
            generator_optimizer.step()

            discriminator_loss_total += discriminator_loss.item()
            generator_loss_total += generator_loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("data_loader did not yield any batches")

        epoch_metrics = {
            "epoch": epoch + 1,
            "discriminator_loss": discriminator_loss_total / batch_count,
            "generator_loss": generator_loss_total / batch_count,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"discriminator loss: {epoch_metrics['discriminator_loss']:.4f}, "
            f"generator loss: {epoch_metrics['generator_loss']:.4f}"
        )

    model.training_history = history
    return model
