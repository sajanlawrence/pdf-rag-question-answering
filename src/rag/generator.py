import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class RAGGenerator:
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is missing. Add it to the project .env file."
            )

        self.llm = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        if not retrieved_docs:
            return "No relevant documents found."

        context = "\n\n".join(
            document["content"] for document in retrieved_docs
        )

        prompt = f"""Answer the question using only the context below.
If the context does not contain the answer, say that the information
was not found in the provided documents.

Context:
{context}

Question: {query}

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content
