import hashlib
from typing import List
from uuid import NAMESPACE_URL, uuid5

import chromadb
from langchain_core.documents import Document

from .config import CHROMA_DIRECTORY, COLLECTION_NAME


def _make_chunk_id(document: Document, index: int) -> str:
    """Create a repeatable ID so rerunning ingestion updates rather than duplicates."""
    source = str(document.metadata.get("source", "unknown"))
    page = str(document.metadata.get("page", ""))
    content_hash = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()
    identity = f"{source}:{page}:{index}:{content_hash}"
    return str(uuid5(NAMESPACE_URL, identity))


class VectorStore:
    def __init__(
        self,
        persist_directory=CHROMA_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
    ):
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "PDF document chunks for RAG",
                "hnsw:space": "cosine",
            },
        )

    def add_documents(self, documents: List[Document], embeddings) -> List[str]:
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length.")

        ids = [
            _make_chunk_id(document, index)
            for index, document in enumerate(documents)
        ]

        if documents:
            self.collection.upsert(
                ids=ids,
                documents=[document.page_content for document in documents],
                metadatas=[document.metadata for document in documents],
                embeddings=[embedding.tolist() for embedding in embeddings],
            )

        print(f"Upserted {len(ids)} chunks into ChromaDB.")
        return ids

    def get_documents(self, ids: List[str]):
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}

        return self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
