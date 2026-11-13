from .bm25_store import BM25Store
from .embeddings import EmbeddingManager
from .generator import RAGGenerator
from .retriever import HybridRetriever
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()
        self.bm25_store.load()

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            bm25_store=self.bm25_store,
        )
        self.generator = RAGGenerator()

    def query(self, question: str, top_k: int = 5) -> str:
        documents = self.retriever.retrieve(question, top_k=top_k)
        return self.generator.answer(question, documents)
