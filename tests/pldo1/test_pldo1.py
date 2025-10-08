"""Tests files for pldo1."""

from pldo.prompt_generation.gemini_prompt_generator import GeminiPromptGenerator
from pldo.prompt_scoring.base_prompt_scorer import PromptScorer
from pldo.prompt_scoring.gemini_prompt_scorer import GeminiPromptScorer


def test_gemini_generator_creation():
    """Test creating a GeminiPromptGenerator."""
    gen = GeminiPromptGenerator(initial_prompt="object", test=True)
    assert gen.initial_prompt == "object"


def test_gemini_scorer_creation():
    """Test creating a GeminiPromptScorer."""
    sco = GeminiPromptScorer(test=True)
    assert isinstance(sco, PromptScorer)
