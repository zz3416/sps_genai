"""Tests for the Assignment 3 GAN architecture and API helper."""

from __future__ import annotations

import unittest

import torch

from helper_lib.generator import generate_sample_tensors, samples_to_png
from helper_lib.model import (
    MNISTDiscriminator,
    MNISTGAN,
    MNISTGenerator,
    get_model,
)


class GANTests(unittest.TestCase):
    def test_generator_has_required_output_shape(self) -> None:
        generator = MNISTGenerator()
        output = generator(torch.randn(4, 100))

        self.assertEqual(output.shape, (4, 1, 28, 28))
        self.assertGreaterEqual(float(output.detach().min()), -1)
        self.assertLessEqual(float(output.detach().max()), 1)

    def test_discriminator_has_required_output_shape(self) -> None:
        discriminator = MNISTDiscriminator()
        output = discriminator(torch.randn(4, 1, 28, 28))

        self.assertEqual(output.shape, (4, 1))
        self.assertGreaterEqual(float(output.detach().min()), 0)
        self.assertLessEqual(float(output.detach().max()), 1)

    def test_get_model_returns_gan(self) -> None:
        self.assertIsInstance(get_model("MNISTGAN"), MNISTGAN)

    def test_generated_samples_can_be_encoded_as_png(self) -> None:
        samples = generate_sample_tensors(MNISTGenerator(), num_samples=4)
        png = samples_to_png(samples)

        self.assertEqual(samples.shape, (4, 1, 28, 28))
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
