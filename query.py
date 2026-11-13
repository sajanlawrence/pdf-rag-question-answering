from rag.pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer = pipeline.query(question, top_k=5)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
