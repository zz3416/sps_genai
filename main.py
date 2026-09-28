from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding

app = FastAPI(title="Simple Bigram Text Generator")

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
    return {"message": "Simple Bigram Text Generator is running"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)
    return {
        "word": request.word,
        "embedding": embedding
    }