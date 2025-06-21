# embeddings/embedder.py
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    """Devuelve una función de embeddings usando all-MiniLM-L6-v2 (optimizado para similitud de oraciones)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
