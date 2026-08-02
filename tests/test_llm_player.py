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
    assert all(
        item["final_action"] in {"fold", "check_call", "raise"} for item in llm.llm_decision_log
    )
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
    assert result["summary"].loc[0, "total_tokens"] > 0


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


def test_opencode_go_provider_uses_local_cli_with_schema_prompt() -> None:
    from reflexive_poker.llm_player import OpenCodeGoProvider

    prompt = None

    def run(value):
        nonlocal prompt
        prompt = value
        return _FakeResponse.output_text

    provider = OpenCodeGoProvider(run=run)
    response = provider.decide({"legal_actions": ["check_call"], "hand_index": 0})
    assert response.payload["action"] == "check_call"
    assert "JSON schema name: poker_decision" in prompt
    assert '"legal_actions": ["check_call"]' in prompt


def test_opencode_go_provider_parses_json_events_usage_and_cost() -> None:
    from reflexive_poker.llm_player import OpenCodeGoProvider

    def run(_prompt):
        return "\n".join(
            [
                '{"type":"text","sessionID":"ses_test","part":{"text":"```json\\n'
                + _FakeResponse.output_text.replace('"', '\\"')
                + '\\n``` trailing text"}}',
                '{"type":"step_finish","sessionID":"ses_test","modelID":"deepseek-v4-flash","modelVersion":"2026-08-01","part":{"tokens":{"input":12,"output":8,"total":21},"cost":0.0012}}',
            ]
        )

    response = OpenCodeGoProvider(run=run).decide(
        {"legal_actions": ["check_call"], "hand_index": 0}
    )
    assert response.total_tokens == 21
    assert response.cost_usd == 0.0012
    assert response.observed_billed_cost == 0.0012
    assert response.cost_observability == "exact"
    assert response.response_id == "ses_test"
    assert response.actual_model == "deepseek-v4-flash"
    assert response.model_version == "2026-08-01"


def test_opencode_go_provider_counts_cache_tokens_as_input_usage() -> None:
    from reflexive_poker.llm_player import OpenCodeGoProvider

    def run(_prompt):
        return (
            '{"type":"text","part":{"text":"{\\"action\\":\\"check_call\\",\\"raise_scale\\":0.5,\\"confidence\\":0.5,\\"situation_summary\\":\\"x\\",\\"rationale\\":\\"x\\",\\"self_model\\":\\"x\\",\\"opponent_model\\":\\"x\\",\\"risk_flags\\":[],\\"next_step\\":\\"x\\"}"}}\n'
            '{"type":"step_finish","part":{"tokens":{"input":6,"output":7,"cache":{"read":11,"write":13}}}}'
        )

    response = OpenCodeGoProvider(run=run).decide(
        {"legal_actions": ["check_call"], "hand_index": 0}
    )
    assert response.input_tokens == 30
    assert response.output_tokens == 7
    assert response.total_tokens == 37
    assert response.cache_read_tokens == 11
    assert response.cache_write_tokens == 13


def test_opencode_session_export_attests_model_identity(monkeypatch) -> None:
    from reflexive_poker.llm_player import OpenCodeGoProvider

    class Completed:
        returncode = 0
        stdout = (
            "Exporting session: ses_test\\n"
            '{"info":{"model":{"providerID":"opencode-go","id":"deepseek-v4-flash"}}}'
        )

    monkeypatch.setattr("reflexive_poker.llm_player.subprocess.run", lambda *_args, **_kwargs: Completed())
    provider = OpenCodeGoProvider()
    assert provider._export_session_model("ses_test") == "deepseek-v4-flash"


def test_codex_provider_parses_json_events_and_usage() -> None:
    from reflexive_poker.llm_player import DECISION_SCHEMA, CodexProvider

    captured = None

    def run(prompt, schema):
        nonlocal captured
        captured = (prompt, schema)
        return "\n".join(
            [
                '{"type":"thread.started"}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"'
                + _FakeResponse.output_text.replace('"', '\\"')
                + '"}}',
                '{"type":"turn.completed","model":{"id":"gpt-5.6","version":"2026-08-01"},"usage":{"input_tokens":12,"output_tokens":8}}',
            ]
        )

    provider = CodexProvider(run=run)
    response = provider.decide({"legal_actions": ["check_call"], "hand_index": 0})
    assert response.payload["action"] == "check_call"
    assert response.input_tokens == 12
    assert response.output_tokens == 8
    assert response.total_tokens == 20
    assert response.actual_model == "gpt-5.6"
    assert response.model_version == "2026-08-01"
    assert response.cost_usd is None
    assert response.observed_billed_cost is None
    assert response.cost_observability == "unavailable"
    assert captured[1] == DECISION_SCHEMA


def test_codex_cli_selector_is_labelled_when_stream_has_no_model(monkeypatch) -> None:
    from reflexive_poker.llm_player import CodexProvider

    provider = CodexProvider(model="gpt-5.6-luna")
    provider.run = lambda _prompt, _schema: "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"'
            + _FakeResponse.output_text.replace('"', '\\"')
            + '"}}',
            '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":8}}',
        ]
    )

    class Completed:
        returncode = 0
        stdout = "codex-cli 0.146.0"

    monkeypatch.setattr(
        "reflexive_poker.llm_player.subprocess.run", lambda *_args, **_kwargs: Completed()
    )
    response = provider.decide({"legal_actions": ["check_call"], "hand_index": 0})

    assert response.actual_model == "gpt-5.6-luna"
    assert response.model_identity_source == "cli_selected_model"
    assert response.response_id == "thread-1"
    assert response.serving_stack_version == "codex-cli 0.146.0"
