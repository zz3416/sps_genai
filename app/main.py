"""FastAPI entry point for the course application."""

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field

from .bigram_model import BigramModel
from .cnn_service import classify_image
from .embedding_model import calculate_embedding
from .gan_service import generate_image_grid


app = FastAPI(title="Applied Generative AI API")

corpus = [
    (
        "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
        "It tells the story of Edmond Dantes, who is falsely imprisoned and "
        "later seeks revenge."
    ),
    "This is another example sentence.",
    "We are generating text based on bigram probabilities.",
    "Bigram models are simple but effective.",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1, examples=["the"])
    length: int = Field(default=20, ge=1, le=200)


class EmbeddingRequest(BaseModel):
    word: str = Field(min_length=1, examples=["apple"])


@app.get("/")
def read_root():
    return {"message": "Applied Generative AI API is running"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)
    return {
        "word": request.word,
        "embedding": embedding,
    }


@app.post("/classify-image")
async def classify_uploaded_image(image: UploadFile = File(...)):
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        result = classify_image(image_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {
        "filename": image.filename,
        **result,
    }


@app.post(
    "/generate-images",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "A PNG grid of generated handwritten digits.",
        },
        503: {"description": "The trained generator is unavailable."},
    },
)
def generate_mnist_images(
    num_samples: int = Query(default=10, ge=1, le=100),
) -> Response:
    """Generate a PNG grid containing synthetic MNIST-style digits."""

    try:
        image_bytes = generate_image_grid(num_samples)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return Response(content=image_bytes, media_type="image/png")
