import pickle
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from .config import BM25_INDEX_PATH


def tokenize(text: str) -> List[str]:
    """Basic tokenizer. Consider a language-aware tokenizer for other languages."""
    return text.lower().split()


class BM25Store:
    def __init__(self, index_path: Path = BM25_INDEX_PATH):
        self.index_path = Path(index_path)
        self.bm25 = None
        self.chunk_ids = []

    def build(self, documents, chunk_ids: List[str]) -> None:
        if len(documents) != len(chunk_ids):
            raise ValueError("Documents and chunk IDs must have the same length.")
        if not documents:
            raise ValueError("Cannot build BM25 index with no documents.")

        tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunk_ids = list(chunk_ids)
        self.save()
        print(f"Built and saved BM25 index for {len(self.chunk_ids)} chunks.")

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as file:
            pickle.dump(
                {"bm25": self.bm25, "chunk_ids": self.chunk_ids},
                file,
            )

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_path}. Run ingestion first."
            )

        with self.index_path.open("rb") as file:
            payload = pickle.load(file)

        self.bm25 = payload["bm25"]
        self.chunk_ids = payload["chunk_ids"]

    def search(self, query: str, top_k: int = 10) -> List[str]:
        if self.bm25 is None:
            self.load()

        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        # Ignore zero-score documents: they have no lexical match.
        matching_indices = [
            index for index in ranked_indices if scores[index] > 0
        ][:top_k]

        return [self.chunk_ids[index] for index in matching_indices]
