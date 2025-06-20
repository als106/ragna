# embeddings/embedder.py
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    """Devuelve una función de embeddings configurada con un modelo optimizado para QA multilingüe."""
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
