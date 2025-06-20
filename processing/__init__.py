# processing/__init__.py
from .splitter import split_documents
from .metadata import calculate_chunk_ids

__all__ = ["split_documents", "calculate_chunk_ids"]
