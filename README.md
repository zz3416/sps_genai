# SPS Generative AI API

This FastAPI project contains the course text-generation, word-embedding, and
CIFAR-10 image-classification endpoints, and an MNIST GAN image generator.

## Assignment 3 GAN architecture

The generator maps a 100-value noise vector through a fully connected layer,
reshapes it to `128 x 7 x 7`, and upsamples twice to produce a grayscale
`28 x 28` image. The discriminator applies the required two convolutional
blocks and returns a single real/fake probability.

Train the GAN on MNIST and save its generator checkpoint and preview image:

```bash
uv run python -m app.train_gan --epochs 10
```

After training, start the API and request a PNG containing generated digits:

```bash
curl -X POST "http://127.0.0.1:8000/generate-images?num_samples=10" \
  --output generated-digits.png
```

## CNN architecture

The Assignment 2 classifier accepts 64x64 RGB images and uses the required
architecture:

1. 3x3 convolution with 16 filters, stride 1, padding 1
2. ReLU and 2x2 max pooling
3. 3x3 convolution with 32 filters, stride 1, padding 1
4. ReLU and 2x2 max pooling
5. Flatten, a 100-unit fully connected layer, ReLU, and a 10-unit output layer

## Train the classifier

```bash
uv run python -m app.train_cnn
```

The command downloads CIFAR-10, resizes its images to 64x64, trains for 10
epochs, evaluates the model, and saves `models/cnn_cifar10.pth`.

## Run the API

```bash
uv run fastapi dev app/main.py
```

Open `http://127.0.0.1:8000/docs`. Use `POST /classify-image` to classify an
uploaded image or `POST /generate-images` to generate MNIST-style digits.

Example:

```bash
curl -X POST http://127.0.0.1:8000/classify-image \
  -F "image=@example.png"
```

## Docker

```bash
docker build -t sps-genai .
docker run --rm -p 8000:80 sps-genai
```
