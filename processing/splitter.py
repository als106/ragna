# processing/splitter.py

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents: list[Document]) -> list[Document]:
    chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", " "],
        length_function=len
    )

    for doc in documents:
        if doc.metadata.get("type") == "qa_pair":
            chunks.append(doc)
        else:
            split_parts = splitter.split_documents([doc])
            chunks.extend(split_parts)

    return chunks
