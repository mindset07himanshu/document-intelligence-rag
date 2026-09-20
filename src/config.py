from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance: float
    max_history_turns: int
    data_dir: str = "data"
    persist_dir: str = "storage/chroma"

    @property
    def together_base_url(self) -> str:
        return "https://api.together.xyz/v1"

def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("TOGETHER_API_KEY", ""),
        model=os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "5")),
        min_relevance=float(os.getenv("MIN_RELEVANCE", "0.42")),
        max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "6")),
    )
