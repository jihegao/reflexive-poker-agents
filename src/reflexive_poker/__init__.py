"""Reflexive Poker Agents research demo."""

from .llm_evaluation import LLMEvaluationConfig, run_llm_evaluation
from .llm_player import (
    CodexProvider,
    DeterministicNarrativeProvider,
    LLMPlayer,
    OpenAIResponsesProvider,
    OpenCodeGoProvider,
)
from .phase1_experiment import Phase1ExperimentConfig, run_phase1_experiment
from .phase1_models import OpponentBeliefState, ProviderBudget, ReasoningTreatment
from .phase1_resumable import (
    FullSimulationRunConfig,
    LLMConfirmationRunConfig,
    run_full_simulation_matrix,
    run_llm_confirmation_resumable,
)
from .type_matchup_experiment import TypeMatchupConfig, run_type_matchups

__all__ = [
    "CodexProvider",
    "DeterministicNarrativeProvider",
    "FullSimulationRunConfig",
    "LLMConfirmationRunConfig",
    "LLMEvaluationConfig",
    "LLMPlayer",
    "OpenAIResponsesProvider",
    "OpenCodeGoProvider",
    "OpponentBeliefState",
    "Phase1ExperimentConfig",
    "ProviderBudget",
    "ReasoningTreatment",
    "TypeMatchupConfig",
    "run_full_simulation_matrix",
    "run_llm_confirmation_resumable",
    "run_llm_evaluation",
    "run_phase1_experiment",
    "run_type_matchups",
]
__version__ = "0.5.0"
