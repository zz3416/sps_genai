"""Model factory for the neural-network, VAE, and GAN activities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    """A fully connected classifier for 3 x 32 x 32 CIFAR-10 images."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, inputs):
        inputs = self.flatten(inputs)
        inputs = F.relu(self.fc1(inputs))
        return self.fc2(inputs)


class CNN(nn.Module):
    """The SimpleCNN architecture from Module 4 Practical 3."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, inputs):
        inputs = self.pool(F.relu(self.conv1(inputs)))
        inputs = self.pool(F.relu(self.conv2(inputs)))
        inputs = inputs.reshape(inputs.size(0), 32 * 8 * 8)
        inputs = F.relu(self.fc1(inputs))
        return self.fc2(inputs)


class EnhancedCNN(nn.Module):
    """The enhanced architecture from Module 4 Practical 3."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, inputs):
        inputs = self.pool(F.relu(self.bn1(self.conv1(inputs))))
        inputs = self.pool(F.relu(self.bn2(self.conv2(inputs))))
        inputs = self.pool(F.relu(self.bn3(self.conv3(inputs))))
        inputs = self.pool(F.relu(self.bn4(self.conv4(inputs))))
        inputs = inputs.reshape(inputs.size(0), 128 * 2 * 2)
        inputs = self.dropout(F.relu(self.fc1(inputs)))
        return self.fc2(inputs)


class VariationalEncoder(nn.Module):
    """Encode a 1 x 32 x 32 image as a Gaussian latent distribution."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, inputs):
        inputs = F.relu(self.conv1(inputs))
        inputs = F.relu(self.conv2(inputs))
        inputs = F.relu(self.conv3(inputs))
        inputs = self.flatten(inputs)
        return self.fc_mu(inputs), self.fc_logvar(inputs)


class Decoder(nn.Module):
    """Decode latent points into 1 x 32 x 32 grayscale images."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.convtrans1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.convtrans2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.convtrans3 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, latent):
        latent = self.fc(latent)
        latent = latent.view(-1, 128, 4, 4)
        latent = F.relu(self.convtrans1(latent))
        latent = F.relu(self.convtrans2(latent))
        return torch.sigmoid(self.convtrans3(latent))


class VAE(nn.Module):
    """Variational autoencoder used in the Module 5 practical."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, inputs):
        mu, logvar = self.encoder(inputs)
        latent = self.reparameterize(mu, logvar)
        return self.decoder(latent), mu, logvar


class Critic(nn.Module):
    """WGAN critic for normalized 3 x 64 x 64 images."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
        )

    def forward(self, images):
        return self.network(images)


class Generator(nn.Module):
    """WGAN generator that maps latent vectors to 3 x 64 x 64 images."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 3, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, latent):
        if latent.ndim == 2:
            latent = latent.view(latent.size(0), self.latent_dim, 1, 1)
        return self.network(latent)


class GAN(nn.Module):
    """Container holding the generator and critic used by the Module 6 WGAN."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim)
        self.critic = Critic()
        self.training_history = []

    def forward(self, latent):
        return self.generator(latent)


class MNISTGenerator(nn.Module):
    """Assignment 3 generator: (batch, 100) noise to 1 x 28 x 28 images."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.fully_connected = nn.Linear(latent_dim, 7 * 7 * 128)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, noise):
        if noise.ndim != 2 or noise.size(1) != self.latent_dim:
            raise ValueError(
                f"Expected noise shape (batch, {self.latent_dim}), "
                f"received {tuple(noise.shape)}"
            )
        features = self.fully_connected(noise)
        features = features.view(noise.size(0), 128, 7, 7)
        return self.network(features)


class MNISTDiscriminator(nn.Module):
    """Assignment 3 discriminator for 1 x 28 x 28 MNIST images."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.network(images)


class MNISTGAN(nn.Module):
    """Container for the Assignment 3 generator and discriminator."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = MNISTGenerator(latent_dim=latent_dim)
        self.discriminator = MNISTDiscriminator()
        self.training_history = []

    def forward(self, noise):
        return self.generator(noise)


def get_model(model_name, num_classes=10, latent_dim=None):
    """Return a classifier, VAE, or GAN model by name."""
    models = {
        "FCNN": FCNN,
        "CNN": CNN,
        "ENHANCEDCNN": EnhancedCNN,
        "VAE": VAE,
        "GAN": GAN,
        "MNISTGAN": MNISTGAN,
    }
    key = model_name.replace("_", "").upper()
    if key not in models:
        choices = ", ".join(
            ("FCNN", "CNN", "EnhancedCNN", "VAE", "GAN", "MNISTGAN")
        )
        raise ValueError(f"Unknown model {model_name!r}. Choose one of: {choices}")
    if key == "VAE":
        return models[key](latent_dim=2 if latent_dim is None else latent_dim)
    if key == "GAN":
        return models[key](latent_dim=100 if latent_dim is None else latent_dim)
    if key == "MNISTGAN":
        return models[key](latent_dim=100 if latent_dim is None else latent_dim)
    return models[key](num_classes=num_classes)
