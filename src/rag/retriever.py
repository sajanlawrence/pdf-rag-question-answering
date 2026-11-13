from collections import defaultdict
from typing import Dict, List

from .bm25_store import BM25Store
from .config import DENSE_CANDIDATES, RRF_K, SPARSE_CANDIDATES
from .embeddings import EmbeddingManager
from .vector_store import VectorStore


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = RRF_K,
) -> List[tuple]:
    """Fuse ranked ID lists using Reciprocal Rank Fusion."""
    fused_scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, document_id in enumerate(ranked_list, start=1):
            fused_scores[document_id] += 1 / (k + rank)

    return sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        bm25_store: BM25Store,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.bm25_store = bm25_store

    def dense_search(self, query: str, top_k: int = DENSE_CANDIDATES) -> List[str]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.vector_store.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return results["ids"][0] if results["ids"] else []

    def sparse_search(self, query: str, top_k: int = SPARSE_CANDIDATES) -> List[str]:
        return self.bm25_store.search(query, top_k=top_k)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = DENSE_CANDIDATES,
        sparse_k: int = SPARSE_CANDIDATES,
    ) -> List[Dict]:
        dense_ids = self.dense_search(query, dense_k)
        sparse_ids = self.sparse_search(query, sparse_k)

        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
        final_ids = [document_id for document_id, _ in fused[:top_k]]

        stored = self.vector_store.get_documents(final_ids)
        records_by_id = {
            document_id: {
                "id": document_id,
                "content": content,
                "metadata": metadata or {},
            }
            for document_id, content, metadata in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            )
        }

        # Preserve RRF order, and expose the fusion score for inspection.
        results = []
        for document_id, rrf_score in fused[:top_k]:
            record = records_by_id.get(document_id)
            if record:
                record["rrf_score"] = rrf_score
                results.append(record)

        return results
