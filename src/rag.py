from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from .config import Settings
from .prompts import SYSTEM_PROMPT

@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    confidence_label: str

class InsuranceRAG:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            collection_name="document_knowledge",
            embedding_function=self.embeddings,
            persist_directory=settings.persist_dir,
        )
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.together_base_url,
        )

    @property
    def document_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def _retrieve(self, question: str):
        return self.vectorstore.similarity_search_with_relevance_scores(
            question,
            k=self.settings.top_k,
        )

    @staticmethod
    def _format_history(history):
        if not history:
            return "No previous conversation."
        return "\n".join(
            f"{role}: {content}" for role, content in history[-12:]
        )

    @staticmethod
    def _source_label(doc) -> str:
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page")
        page_label = f", page {int(page) + 1}" if isinstance(page, int) else ""
        return f"{source}{page_label}"

    def answer(self, question: str, history=None) -> RAGResult:
        if not self.settings.api_key:
            return RAGResult(
                "Add TOGETHER_API_KEY to your .env file before asking the model a question.",
                [],
                "not configured",
            )

        matches = self._retrieve(question)
        strong = [
            (doc, score)
            for doc, score in matches
            if score >= self.settings.min_relevance
        ]

        if not strong:
            return RAGResult(
                "I couldn't find enough relevant information in the uploaded documents to answer that reliably.",
                [],
                "low",
            )

        context = "\n\n---\n\n".join(
            f"[Source: {self._source_label(doc)}]\n{doc.page_content}"
            for doc, _ in strong
        )
        sources = list(dict.fromkeys(self._source_label(doc) for doc, _ in strong))

        prompt = SYSTEM_PROMPT.format(
            history=self._format_history(history or []),
            context=context,
            question=question,
        )

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=350,
        )

        answer = response.choices[0].message.content.strip()
        top_score = max(score for _, score in strong)
        confidence = "high" if top_score >= 0.65 else "moderate"

        return RAGResult(answer, sources, confidence)
