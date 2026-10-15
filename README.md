# SPS Generative AI API

This FastAPI project contains the course text-generation, word-embedding, and
CIFAR-10 image-classification endpoints.

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

Open `http://127.0.0.1:8000/docs` and use `POST /classify-image` to upload an
image. The response contains its predicted CIFAR-10 class and confidence.

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
