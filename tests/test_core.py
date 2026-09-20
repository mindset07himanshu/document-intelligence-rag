from src.config import get_settings
from src.prompts import SYSTEM_PROMPT

def test_default_settings_are_sane():
    settings = get_settings()
    assert settings.chunk_size > settings.chunk_overlap
    assert settings.top_k > 0
    assert 0 <= settings.min_relevance <= 1

def test_prompt_requires_grounding():
    assert "Use only the supplied context" in SYSTEM_PROMPT
    assert "Do not invent" in SYSTEM_PROMPT
