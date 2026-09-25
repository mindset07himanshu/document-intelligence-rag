"""
Benchmark the local RAG pipeline.

Usage:
    python -m eval.benchmark

The script measures:
- number of PDF files in data/
- number of indexed chunks
- retrieval latency for each evaluation question
- total answer latency when a valid Together API key is configured

It prints measured values; it does not invent or hard-code latency numbers.
"""

import statistics
import time
from pathlib import Path

from src.config import get_settings
from src.ingestion import build_index
from src.rag import InsuranceRAG

QUESTIONS = [
    "What is risk management?",
    "What is indemnity?",
    "Under what circumstances can a person withdraw money from their MPF account?",
]


def main():
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    pdf_count = len(list(data_dir.glob("*.pdf")))

    print(f"Test PDF count: {pdf_count}")

    chunks = build_index()
    print(f"Indexed chunks: {chunks}")

    agent = InsuranceRAG(settings)

    retrieval_times = []
    for question in QUESTIONS:
        start = time.perf_counter()
        matches = agent._retrieve(question)
        elapsed_ms = (time.perf_counter() - start) * 1000
        retrieval_times.append(elapsed_ms)
        print(f"Retrieval | {elapsed_ms:.2f} ms | {question} | matches={len(matches)}")

    print(f"Average retrieval latency: {statistics.mean(retrieval_times):.2f} ms")

    if not settings.api_key:
        print("Total answer latency: not measured (TOGETHER_API_KEY is not configured).")
        return

    answer_times = []
    for question in QUESTIONS:
        start = time.perf_counter()
        result = agent.answer(question)
        elapsed_ms = (time.perf_counter() - start) * 1000
        answer_times.append(elapsed_ms)
        print(
            f"Total answer | {elapsed_ms:.2f} ms | "
            f"confidence={result.confidence_label} | {question}"
        )

    print(f"Average total answer latency: {statistics.mean(answer_times):.2f} ms")


if __name__ == "__main__":
    main()
