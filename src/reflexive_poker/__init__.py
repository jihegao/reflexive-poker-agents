"""Reflexive Poker Agents research demo."""

from .experiment import ExperimentConfig, run_ablation
from .image_shaping_experiment import ImageShapingConfig, run_image_shaping
from .llm_evaluation import LLMEvaluationConfig, run_llm_evaluation
from .llm_player import DeterministicNarrativeProvider, LLMPlayer, OpenAIResponsesProvider
from .type_matchup_experiment import TypeMatchupConfig, run_type_matchups

__all__ = [
    "ExperimentConfig",
    "ImageShapingConfig",
    "LLMEvaluationConfig",
    "TypeMatchupConfig",
    "LLMPlayer",
    "DeterministicNarrativeProvider",
    "OpenAIResponsesProvider",
    "run_ablation",
    "run_image_shaping",
    "run_llm_evaluation",
    "run_type_matchups",
]
__version__ = "0.5.0"
