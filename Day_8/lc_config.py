
"""Day 8 configuration: chat model, embeddings, and Chroma."""

import hashlib
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PROVIDER = os.getenv("PROVIDER", "ollama").strip().lower()
EMBED_PROVIDER = os.getenv(
    "EMBED_PROVIDER",
    "ollama" if PROVIDER == "ollama" else "fastembed",
).strip().lower()

DB_DIR = str(BASE_DIR / "chroma_db")
DATA_DIR = BASE_DIR / "data"
COLLECTION = "college_helpdesk"


def get_model(temperature=0):
    """Return the chat model selected in .env."""
    model_name = os.getenv(
        "MODEL",
        "qwen2.5:1.5b" if PROVIDER == "ollama"
        else "openai/gpt-oss-20b",
    )

    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI

    if PROVIDER == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY in your local .env file.")

        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=model_name,
            temperature=temperature,
        )

    if PROVIDER == "huggingface":
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise RuntimeError("Set HF_TOKEN in your local .env file.")

        return ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
            model=model_name,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported PROVIDER: {PROVIDER}")


class HashEmbeddings(Embeddings):
    """Offline keyword-based fallback; not a semantic embedding model."""

    STOP = set(
        "a an the is are was to of in on for and or by at be it "
        "this that with what how do does i my can".split()
    )

    def _vec(self, text):
        vector = [0.0] * 256

        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in self.STOP:
                continue

            # Basic normalization: fees -> fee, rules -> rule.
            if word.endswith("s") and len(word) > 3:
                word = word[:-1]

            index = int(
                hashlib.md5(word.encode("utf-8")).hexdigest(), 16
            ) % len(vector)

            vector[index] += 1.0

        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts):
        return [self._vec(text) for text in texts]

    def embed_query(self, text):
        return self._vec(text)


def get_embeddings():
    """Return the selected embedding implementation."""
    if EMBED_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("EMBED_MODEL", "nomic-embed-text")
        )

    if EMBED_PROVIDER == "fastembed":
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(
            model_name=os.getenv(
                "EMBED_MODEL", "BAAI/bge-small-en-v1.5"
            )
        )

    if EMBED_PROVIDER == "hash":
        return HashEmbeddings()

    raise ValueError(
        f"Unsupported EMBED_PROVIDER: {EMBED_PROVIDER}"
    )


def get_vectorstore():
    """Open the persistent Chroma vector store."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
