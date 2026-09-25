import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Please add it to your .env file."
    )


# ---------------------------------------------------------
# Gemini client
# ---------------------------------------------------------

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ---------------------------------------------------------
# Gemini embedding model
# ---------------------------------------------------------

EMBEDDING_MODEL = "gemini-embedding-001"


# ---------------------------------------------------------
# Split study material into chunks
# ---------------------------------------------------------

def chunk_text(
    text,
    chunk_size=1200,
    overlap=200
):

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ---------------------------------------------------------
# Create embeddings for document chunks
# ---------------------------------------------------------

def create_embeddings(chunks):

    if not chunks:
        return []

    embeddings = []

    for chunk in chunks:

        result = client.models.embed_content(

            model=EMBEDDING_MODEL,

            contents=chunk,

            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )

        )

        if not result.embeddings:
            continue

        embedding = result.embeddings[0].values

        embeddings.append(embedding)

    return embeddings


# ---------------------------------------------------------
# Calculate cosine similarity
# ---------------------------------------------------------

def cosine_similarity(
    vector_a,
    vector_b
):

    a = np.array(
        vector_a,
        dtype=np.float32
    )

    b = np.array(
        vector_b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ---------------------------------------------------------
# Retrieve the most relevant chunks
# ---------------------------------------------------------

def retrieve_chunks(
    chunks,
    embeddings,
    query,
    top_k=5
):

    if not chunks or not embeddings:

        return []


    # ---------------------------------------------
    # Create query embedding
    # ---------------------------------------------

    query_result = client.models.embed_content(

        model=EMBEDDING_MODEL,

        contents=query,

        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )

    )


    if not query_result.embeddings:

        return []


    query_embedding = (
        query_result
        .embeddings[0]
        .values
    )


    # ---------------------------------------------
    # Calculate similarity
    # ---------------------------------------------

    scored_chunks = []

    for index, embedding in enumerate(
        embeddings
    ):

        # Make sure we don't access a
        # chunk that doesn't have an embedding.
        if index >= len(chunks):
            break

        score = cosine_similarity(

            query_embedding,

            embedding

        )

        scored_chunks.append({

            "chunk": chunks[index],

            "score": score

        })


    # ---------------------------------------------
    # Sort highest similarity first
    # ---------------------------------------------

    scored_chunks.sort(

        key=lambda item: item["score"],

        reverse=True

    )


    return scored_chunks[:top_k]


# ---------------------------------------------------------
# Build context for Gemini
# ---------------------------------------------------------

def build_context(
    retrieved_chunks
):

    if not retrieved_chunks:

        return ""


    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            item["chunk"]
        )


    return "\n\n---\n\n".join(
        context_parts
    )