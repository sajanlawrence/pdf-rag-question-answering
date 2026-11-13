from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIRECTORY


def load_pdfs(pdf_directory: Path = PDF_DIRECTORY) -> List[Document]:
    """Load every PDF in a directory into LangChain Document objects."""
    pdf_directory = Path(pdf_directory)

    if not pdf_directory.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_directory}")

    pdf_files = sorted(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_directory}. Add PDFs and run ingestion again."
        )

    all_documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        try:
            documents = PyPDFLoader(str(pdf_file)).load()
            for document in documents:
                document.metadata["source"] = pdf_file.name
                document.metadata["file_type"] = "pdf"
            all_documents.extend(documents)
            print(f"  Loaded {len(documents)} pages")
        except Exception as exc:
            print(f"  Failed to load {pdf_file.name}: {exc}")

    if not all_documents:
        raise RuntimeError("No PDF pages could be loaded.")

    return all_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split loaded pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages.")
    return chunks
