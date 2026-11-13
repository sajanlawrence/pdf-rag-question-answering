# Hybrid Search RAG (PDF Question Answering)

A modular PDF-based RAG project using:
- Sentence Transformers for dense embeddings
- ChromaDB for vector storage and cosine-distance retrieval
- BM25 for sparse keyword retrieval
- Reciprocal Rank Fusion (RRF) to combine retrieval rankings
- Groq Chat for answer generation

## Project structure

```text
hybrid_rag_project/
├── data/
│   ├── pdfs/                 # Put source PDFs here
│   ├── chroma_db/            # Created during ingestion
│   └── bm25_index.pkl        # Created during ingestion
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── config.py
│       ├── ingestion.py
│       ├── embeddings.py
│       ├── vector_store.py
│       ├── bm25_store.py
│       ├── retriever.py
│       ├── generator.py
│       └── pipeline.py
├── ingest.py
├── query.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate      # Windows

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_actual_key
```

Add your PDF files to `data/pdfs/`.

## Build the indexes

Run this whenever you add or replace PDFs:

```bash
PYTHONPATH=src python ingest.py
```

This loads PDFs, chunks the text, generates embeddings, upserts chunks into ChromaDB,
and rebuilds the BM25 index with matching chunk IDs.

## Ask questions

```bash
PYTHONPATH=src python query.py
```

The retriever runs dense search and BM25 search, fuses the ranked chunk IDs with RRF,
and sends the final retrieved chunks to the LLM.

## Important notes

- Run ingestion before querying.
- The BM25 pickle file must be kept aligned with the ChromaDB collection. Rebuild both
  indexes together by running `ingest.py` after changing the PDF corpus.
- Only load pickle files that you created and trust. Unpickling untrusted files can
  execute malicious code.
- The BM25 tokenizer here uses lowercase whitespace tokenization as a simple baseline.
  For production, consider better tokenization, normalization, stemming/lemmatization,
  and language-specific handling.
- The current RRF implementation uses rank positions, not raw vector/BM25 scores.
- This project is a local single-process baseline; for larger corpora, consider a
  persistent sparse-search engine and retrieval evaluation.
