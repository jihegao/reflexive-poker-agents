from pathlib import Path

from reflexive_poker.agents import AgentStyle
from reflexive_poker.environment import EnvironmentConfig, HoldemEnvironment
from reflexive_poker.llm_evaluation import LLMEvaluationConfig, run_llm_evaluation
from reflexive_poker.llm_player import DeterministicNarrativeProvider, LLMPlayer
from reflexive_poker.tournament_agents import make_tournament_agent


def test_llm_player_records_decisions_and_reflections() -> None:
    llm = LLMPlayer(
        "llm",
        1,
        DeterministicNarrativeProvider(seed=2),
        AgentStyle(equity_samples=1),
    )
    tag = make_tournament_agent("tag", "tag", ("llm",), 3, equity_samples=1)
    records = HoldemEnvironment(
        [llm, tag], seed=4, config=EnvironmentConfig(regime_switch_hand=99)
    ).play(3)
    assert len(records) == 3
    assert llm.llm_decision_log
    assert len(llm.llm_reflection_log) == 3
    assert all(item["final_action"] in {"fold", "check_call", "raise"} for item in llm.llm_decision_log)
    assert all(item["output"]["rationale"] for item in llm.llm_decision_log if item["output"])


def test_small_llm_evaluation(tmp_path: Path) -> None:
    result = run_llm_evaluation(
        LLMEvaluationConfig(
            provider="mock",
            opponents=("tag",),
            seeds=(11,),
            hands_per_mirror=3,
            equity_samples=1,
            output_dir=tmp_path,
        )
    )
    assert len(result["matches"]) == 2
    assert (tmp_path / "decision_traces.jsonl.gz").exists()
    assert (tmp_path / "reflection_traces.jsonl.gz").exists()
    assert (tmp_path / "trace_examples.md").exists()

class _FakeUsage:
    input_tokens = 12
    output_tokens = 8
    total_tokens = 20


class _FakeResponse:
    id = "resp_test"
    usage = _FakeUsage()
    output_text = '{"action":"check_call","raise_scale":0.5,"confidence":0.7,"situation_summary":"test","rationale":"test","self_model":"test","opponent_model":"test","risk_flags":[],"next_step":"test"}'


class _FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse()


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


def test_openai_provider_uses_responses_structured_output() -> None:
    from reflexive_poker.llm_player import OpenAIResponsesProvider

    client = _FakeClient()
    provider = OpenAIResponsesProvider(model="gpt-5-mini", client=client)
    response = provider.decide({"legal_actions": ["check_call"], "hand_index": 0})
    assert response.payload["action"] == "check_call"
    assert response.total_tokens == 20
    assert client.responses.kwargs["text"]["format"]["type"] == "json_schema"
    assert client.responses.kwargs["text"]["format"]["strict"] is True
