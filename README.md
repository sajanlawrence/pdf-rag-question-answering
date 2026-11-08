# PDF-Based RAG Question Answering System

A Retrieval-Augmented Generation (RAG) application built from scratch using Python that allows users to ask questions about PDF documents and receive context-aware answers using Large Language Models (LLMs).

The application processes PDF documents, converts their content into vector embeddings, stores them in ChromaDB, retrieves relevant information using semantic similarity search, and generates answers using Groq.

## Project Overview

Large language models do not automatically have access to information contained in private or custom PDF documents.

This project addresses that problem using Retrieval-Augmented Generation (RAG).

Instead of relying entirely on the LLM's pretrained knowledge, the application retrieves relevant document content and provides it as context to the LLM before generating an answer.

## Architecture

```text
             PDF Documents
                   |
                   v
             PDF Loading
                   |
                   v
             Text Extraction
                   |
                   v
              Text Splitting
                   |
                   v
           Generate Embeddings
                   |
                   v
          Store in ChromaDB
                   |
                   |
              User Query
                   |
                   v
           Generate Query Embedding
                   |
                   v
          Semantic Similarity Search
                   |
                   v
           Retrieve Relevant Chunks
                   |
                   v
          Build Context for LLM
                   |
                   v
           Groq LLM (GPT-OSS)
                   |
                   v
             Generated Answer
```

## Features

- **PDF Ingestion:** Load and extract text from multiple PDF documents.
- **Text Chunking:** Split extracted content into smaller, manageable chunks using RecursiveCharacterTextSplitter.
- **Embedding Generation:** Generate vector embeddings using Sentence Transformers.
- **Vector Database:** Store and persist document embeddings using ChromaDB.
- **Semantic Search:** Retrieve relevant document chunks based on query similarity.
- **Cosine Similarity:** Use cosine distance for semantic similarity retrieval.
- **Top-K Retrieval:** Retrieve the most relevant document chunks for a given query.
- **Similarity Threshold:** Filter retrieved results based on a configurable similarity score.
- **LLM Integration:** Use Groq to generate natural-language answers based on retrieved document context.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | PDF loading, text splitting and LLM integration |
| PyPDF | PDF document processing |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Persistent vector database |
| Groq | LLM inference |
| GPT-OSS-120B | Language model |

## Project Structure

```text
pdf-rag-project/
│
├── data/
│   └── pdfs/
│       ├── 01_company_overview.pdf
│       ├── 02_people_and_workplace_policy.pdf
│       ├── 03_it_security_and_data_handling.pdf
│       ├── 04_travel_expense_and_procurement.pdf
│       └── 05_products_customers_and_support_playbook.pdf
│
├── notebook/
│   └── pdf_rag_pipeline.ipynb
│
├── .env
├── .gitignore
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sajanlawrence/pdf-rag-question-answering.git

cd pdf-rag-question-answering
```

### 2. Install dependencies

If you have `uv` installed:

```bash
uv sync
```

Alternatively, using pip:

```bash
pip install -r requirements.txt
```

### 3. Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

You can obtain an API key from [Groq Console](https://console.groq.com/).

**Important:** Never commit your `.env` file or expose your API key publicly.

### 4. Run the notebook

Open the project in VS Code.

Navigate to:

```text
notebook/pdf_rag_pipeline.ipynb
```

Select the Python environment where the dependencies are installed and execute the notebook cells in sequence.

## How It Works

### 1. Document Processing

PDF documents are loaded and their textual content is extracted using LangChain's PDF loading functionality.

### 2. Text Chunking

The extracted text is divided into smaller chunks using RecursiveCharacterTextSplitter.

Chunk size and overlap help preserve relevant context while preparing documents for embedding generation.

### 3. Embedding Generation

The Sentence Transformers model `all-MiniLM-L6-v2` converts text chunks into numerical vector representations.

These embeddings capture semantic information and allow the application to identify contextually relevant content.

### 4. Vector Storage

The generated embeddings are stored in a persistent ChromaDB collection.

The collection uses cosine distance for similarity-based retrieval.

### 5. Query Processing and Retrieval

When a user submits a question:

1. The question is converted into an embedding.
2. ChromaDB searches for the most relevant document chunks.
3. The top-K matching chunks are retrieved.
4. A similarity threshold is applied to filter results.

### 6. Answer Generation

The retrieved document content is passed to the Groq-hosted GPT-OSS-120B model as context.

The LLM uses the retrieved information to generate a natural-language response.

## Future Improvements

- Add source citations with PDF filenames and page numbers.
- Implement conversational memory for follow-up questions.
- Experiment with different embedding models.
- Add hybrid search combining keyword and semantic retrieval.
- Implement a user interface using Streamlit or FastAPI.
- Evaluate retrieval quality using metrics such as Precision@K and Recall@K.
- Explore reranking techniques to improve retrieval relevance.

## Learning Outcomes

Through this project, I explored and implemented the core components of a RAG pipeline:

- Understanding Retrieval-Augmented Generation.
- PDF ingestion and document preprocessing.
- Text chunking strategies.
- Vector embeddings and semantic search.
- Cosine similarity and distance metrics.
- Persistent vector storage using ChromaDB.
- Connecting an LLM to retrieved external knowledge.

## Author

**Sajan Lawrence**

GitHub: https://github.com/sajanlawrence

---

*Built as a hands-on project to understand and implement Retrieval-Augmented Generation from scratch.*