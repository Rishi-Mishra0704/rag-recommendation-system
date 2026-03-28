import psycopg2
import requests
from pgvector.psycopg2 import register_vector

from recommendation.constants.constants import DB_CONFIG, OLLAMA_EMBED_URL, EMBED_MODEL


def connect_db() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]
