"""Reflexive Poker Agents research demo."""

from .llm_evaluation import LLMEvaluationConfig, run_llm_evaluation
from .llm_player import (
    CodexProvider,
    DeterministicNarrativeProvider,
    LLMPlayer,
    OpenAIResponsesProvider,
    OpenCodeGoProvider,
)
from .type_matchup_experiment import TypeMatchupConfig, run_type_matchups

__all__ = [
    "CodexProvider",
    "DeterministicNarrativeProvider",
    "LLMEvaluationConfig",
    "LLMPlayer",
    "OpenAIResponsesProvider",
    "OpenCodeGoProvider",
    "TypeMatchupConfig",
    "run_llm_evaluation",
    "run_type_matchups",
]
__version__ = "0.5.0"
