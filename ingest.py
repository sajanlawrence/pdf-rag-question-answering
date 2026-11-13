from rag.bm25_store import BM25Store
from rag.config import PDF_DIRECTORY
from rag.embeddings import EmbeddingManager
from rag.ingestion import load_pdfs, split_documents
from rag.vector_store import VectorStore


def main():
    pages = load_pdfs(PDF_DIRECTORY)
    chunks = split_documents(pages)

    embedding_manager = EmbeddingManager()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    vector_store = VectorStore()
    chunk_ids = vector_store.add_documents(chunks, embeddings)

    bm25_store = BM25Store()
    bm25_store.build(chunks, chunk_ids)

    print("\nIngestion completed.")
    print(f"Pages loaded: {len(pages)}")
    print(f"Chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
