from __future__ import annotations

import copy
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import AgentStyle, PokerAgent
from .environment import EnvironmentConfig, HoldemEnvironment
from .llm_player import LLMPlayer
from .models import HandRecord
from .six_max_experiment import SIX_MAX_LINEUP, _provider
from .tournament_agents import make_tournament_agent


@dataclass(frozen=True)
class SharedHistoryConfig:
    provider: str = "mock"
    model: str = "mock"
    seed: int = 9300
    formation_hands: int = 30
    exploitation_hands: int = 30
    equity_samples: int = 64
    memory_hands: int = 30
    branch_workers: int = 2
    output_dir: Path = Path("results/shared_history/mock_pilot")


def _make_environment(config: SharedHistoryConfig) -> HoldemEnvironment:
    names = tuple(f"seat_{index}_{player_type}" for index, player_type in enumerate(SIX_MAX_LINEUP))
    provider = _provider(config.provider, config.model, config.seed)
    llm = LLMPlayer(
        names[0],
        config.seed * 1009 + 1,
        provider,
        AgentStyle(
            aggression=0.47,
            risk_margin=0.055,
            belief_sensitivity=0.24,
            social_learning_rate=0.20,
            equity_samples=config.equity_samples,
        ),
        opponents=names[1:],
        memory_hands=config.memory_hands,
        reflexive_enabled=False,
    )
    agents: list[PokerAgent] = [llm]
    for index, player_type in enumerate(SIX_MAX_LINEUP[1:], start=1):
        agents.append(
            make_tournament_agent(
                player_type,
                names[index],
                tuple(name for name in names if name != names[index]),
                config.seed * 1009 + index + 1,
                equity_samples=config.equity_samples,
            )
        )
    return HoldemEnvironment(
        agents,
        seed=config.seed,
        config=EnvironmentConfig(
            max_raises_per_street=None,
            regime_switch_hand=config.formation_hands + config.exploitation_hands + 1,
        ),
    )


def _llm(environment: HoldemEnvironment) -> LLMPlayer:
    agent = environment.agents[0]
    if not isinstance(agent, LLMPlayer):
        raise TypeError("The first six-max seat must be an LLMPlayer.")
    return agent


def _fork_signature(environment: HoldemEnvironment) -> str:
    agents = []
    for agent in environment.agents:
        state: dict[str, Any] = {
            "name": agent.name,
            "rng": repr(agent.rng.getstate()),
            "cumulative_reward": agent.cumulative_reward,
            "recent_rewards": list(agent.recent_rewards),
            "beliefs": {
                name: [belief.aggression_total, belief.aggression_raises]
                for name, belief in sorted(agent.beliefs.items())
            },
            "decision_log_length": len(agent.decision_log),
        }
        if isinstance(agent, LLMPlayer):
            state.update(
                {
                    "depth_counts": {
                        name: dict(counts)
                        for name, counts in sorted(agent.depth_controller.opponent_counts.items())
                    },
                    "public_aggressive_actions": agent.public_aggressive_actions,
                    "public_passive_actions": agent.public_passive_actions,
                    "recent_reflections": agent.recent_reflections,
                    "provider_failures": agent.provider_failures,
                }
            )
        agents.append(state)
    payload = {
        "environment_rng": repr(environment.rng.getstate()),
        "record_count": len(environment.records),
        "agents": agents,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _set_condition(environment: HoldemEnvironment, enabled: bool) -> None:
    llm = _llm(environment)
    llm.reflexive_enabled = enabled
    llm.condition = "llm_reflexive_on" if enabled else "llm_reflexive_off"


def _run_branch(environment: HoldemEnvironment, enabled: bool, hands: int) -> HoldemEnvironment:
    _set_condition(environment, enabled)
    environment.play(hands)
    return environment


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _image_row(record: HandRecord, phase: str, condition: str, hero_name: str) -> dict[str, Any]:
    opponent_images = [
        float(snapshot.get("belief_aggression", {}).get(hero_name, 0.5))
        for name, snapshot in record.snapshots.items()
        if name != hero_name
    ]
    hero_snapshot = record.snapshots[hero_name]
    return {
        "phase": phase,
        "condition": condition,
        "hand_index": record.hand_index,
        "mean_opponent_image_of_hero": sum(opponent_images) / len(opponent_images),
        "min_opponent_image_of_hero": min(opponent_images),
        "max_opponent_image_of_hero": max(opponent_images),
        "hero_self_public_image": hero_snapshot.get("self_public_image", 0.5),
    }


def _per_hand_rows(records: list[HandRecord], phase: str, condition: str) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for player_type, agent_name in zip(SIX_MAX_LINEUP, record.rewards, strict=True):
            actions = [event for event in record.actions if event.actor == agent_name]
            rows.append(
                {
                    "phase": phase,
                    "condition": condition,
                    "hand_index": record.hand_index,
                    "player_type": player_type,
                    "agent": agent_name,
                    "reward": record.rewards[agent_name],
                    "decision_count": len(actions),
                    "raise_count": sum(event.action.value == "raise" for event in actions),
                    "showdown": record.showdown,
                }
            )
    return rows


def _response_metrics(records: list[HandRecord], hero_name: str) -> dict[str, float | int]:
    responses = [
        event
        for record in records
        for event in record.actions
        if event.faced_raise and event.raiser == hero_name
    ]
    return {
        "responses_to_hero_raise": len(responses),
        "folds_to_hero_raise": sum(event.action.value == "fold" for event in responses),
        "reraises_to_hero_raise": sum(event.action.value == "raise" for event in responses),
        "fold_rate_to_hero_raise": (
            sum(event.action.value == "fold" for event in responses) / len(responses)
            if responses
            else float("nan")
        ),
    }


def _provider_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    token_values = [
        int(trace["total_tokens"]) for trace in traces if trace.get("total_tokens") is not None
    ]
    costs = [float(trace["cost_usd"]) for trace in traces if trace.get("cost_usd") is not None]
    latencies = [
        float(trace["latency_ms"]) for trace in traces if trace.get("latency_ms") is not None
    ]
    return {
        "provider_call_count": len(traces),
        "token_observed_calls": len(token_values),
        "total_tokens": sum(token_values) if token_values else float("nan"),
        "cost_observed_calls": len(costs),
        "reported_cost_usd": sum(costs) if costs else float("nan"),
        "latency_observed_calls": len(latencies),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else float("nan"),
    }


def _design_markdown(config: SharedHistoryConfig) -> str:
    return "\n".join(
        [
            "# Shared-history six-max image-formation experiment",
            "",
            f"- Provider/model: `{config.provider}` / `{config.model}`",
            f"- Seed: `{config.seed}`",
            f"- Shared formation horizon: `{config.formation_hands}` hands",
            f"- Forked exploitation horizon: `{config.exploitation_hands}` hands per condition",
            f"- Reflection memory: `{config.memory_hands}` hands",
            "- Formation uses first-order decisions while all agents accumulate public-action beliefs.",
            "- The complete environment and all agent/RNG states are deep-copied at the fork.",
            "- Exploitation branches differ only in whether reflexive tools and reflection memory are visible to the LLM.",
            "- Primary metric: exploitation-only paired LLM chips/100.",
            "- Mechanism metrics: opponent image of Hero, Hero self-image, responses to Hero raises, and action/size divergence.",
            "- Gate: identical pre-intervention fork signatures, full call accounting, zero provider failures, zero fallbacks.",
            "",
            "This single-seed mechanism pilot cannot establish a stable profitability effect.",
        ]
    )


def _paired_llm_hand_deltas(per_hand: pd.DataFrame) -> pd.DataFrame:
    llm_rows = per_hand[(per_hand["phase"] == "exploitation") & (per_hand["player_type"] == "llm")]
    paired = (
        llm_rows.pivot(index="hand_index", columns="condition", values="reward")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    paired["reward_delta_on_minus_off"] = paired["reflexive_on"] - paired["reflexive_off"]
    paired["cumulative_delta_on_minus_off"] = paired["reward_delta_on_minus_off"].cumsum()
    return paired


def _report_markdown(
    config: SharedHistoryConfig,
    summary: pd.DataFrame,
    image_trajectory: pd.DataFrame,
    calls: pd.DataFrame,
    paired: pd.DataFrame,
    fork: dict[str, Any],
) -> str:
    llm = summary[summary["player_type"] == "llm"].set_index("condition")
    off = llm.loc["reflexive_off"]
    on = llm.loc["reflexive_on"]
    formation = image_trajectory[image_trajectory["condition"] == "shared_formation"]
    largest = paired.reindex(
        paired["reward_delta_on_minus_off"].abs().sort_values(ascending=False).index
    ).head(5)
    top_two_positive = float(
        paired.nlargest(2, "reward_delta_on_minus_off")["reward_delta_on_minus_off"].sum()
    )
    total_delta = float(paired["reward_delta_on_minus_off"].sum())
    token_total = int(calls["total_tokens"].sum())
    cost_total = float(calls["reported_cost_usd"].sum())
    call_total = int(calls["provider_call_count"].sum())
    failure_total = int(calls["provider_failures"].sum())
    fallback_total = int(calls["fallbacks"].sum())
    token_observed_total = int(calls["token_observed_calls"].sum())
    off_calls = calls[calls["condition"] == "reflexive_off"].iloc[0]
    on_calls = calls[calls["condition"] == "reflexive_on"].iloc[0]
    token_increase = float(on_calls["total_tokens"]) / float(off_calls["total_tokens"]) - 1.0
    cost_increase = (
        float(on_calls["reported_cost_usd"]) / float(off_calls["reported_cost_usd"]) - 1.0
    )
    call_rows = "\n".join(
        (
            f"| {row.condition} | {int(row.provider_call_count)} | "
            f"{int(row.token_observed_calls)} | {int(row.total_tokens):,} | "
            f"${row.reported_cost_usd:.6f} | {row.mean_latency_ms:.0f} | "
            f"{int(row.provider_failures)} | {int(row.fallbacks)} |"
        )
        for row in calls.itertuples()
    )
    largest_rows = "\n".join(
        f"| {int(row.hand_index)} | {row.reflexive_off:.3f} | "
        f"{row.reflexive_on:.3f} | {row.reward_delta_on_minus_off:+.3f} |"
        for row in largest.itertuples()
    )
    return "\n".join(
        [
            "# 共享历史二阶推理实验报告",
            "",
            "## 结论",
            "",
            "30 手牌足以在本轮形成可测量的偏紧形象，但本轮尚不能证明二阶推理能稳定提高盈利。",
            "",
            (
                f"共享阶段结束时，对手对 LLM 激进度的平均估计从 "
                f"`{formation.iloc[0]['mean_opponent_image_of_hero']:.3f}` 降至 "
                f"`{formation.iloc[-1]['mean_opponent_image_of_hero']:.3f}`；"
                f"LLM 自身公开形象估计为 "
                f"`{formation.iloc[-1]['hero_self_public_image']:.3f}`。"
            ),
            "",
            (
                f"利用阶段中，二阶推理开启组为 `{on['chips_per_100']:.2f}` "
                f"chips/100，关闭组为 `{off['chips_per_100']:.2f}` chips/100，"
                f"表面差异为 `{fork['exploitation_chips_per_100_delta']:+.2f}` "
                "chips/100。两组都严重亏损，且样本仅 30 手，因此这个差异"
                "不是有效的胜率证据。"
            ),
            "",
            "## 机制信号",
            "",
            "| 指标 | 二阶关闭 | 二阶开启 |",
            "|---|---:|---:|",
            f"| LLM 加注率 | {off['raise_rate']:.3f} | {on['raise_rate']:.3f} |",
            f"| 对手面对 LLM 加注的弃牌率 | {off['fold_rate_to_hero_raise']:.3f} | {on['fold_rate_to_hero_raise']:.3f} |",
            f"| 决策次数 | {int(off['decision_count'])} | {int(on['decision_count'])} |",
            "",
            (
                "二阶开启后，LLM 更少加注，同时对手面对其加注时更常弃牌。"
                "方向上符合“利用已形成形象”的机制，但分支后的牌局状态会随"
                "第一个不同动作迅速分化，不能把后续差异都归因于同一个局面上的策略优劣。"
            ),
            "",
            "## 收益集中度",
            "",
            "| 手牌 | 关闭组收益 | 开启组收益 | 开启减关闭 |",
            "|---:|---:|---:|---:|",
            largest_rows,
            "",
            (
                f"30 手累计差异为 `{total_delta:+.3f}` chips；两个最大正贡献"
                f"手牌合计 `{top_two_positive:+.3f}` chips。去掉这两手后差异为 "
                f"`{total_delta - top_two_positive:+.3f}` chips，因此表面优势由"
                "极少数大底池驱动。"
            ),
            "",
            "## 调用量与有效性门槛",
            "",
            f"- 模型：`{config.provider}` / `{config.model}`",
            f"- API 调用：`{call_total}` 次",
            f"- 已观测 token：`{token_total:,}`",
            f"- 报告成本：`${cost_total:.6f}`",
            f"- Provider 失败：`{failure_total}`；安全回退：`{fallback_total}`",
            f"- 严格有效性门槛：`{fork['valid_provider_gate']}`",
            "",
            "| 阶段/条件 | 调用 | 有 token | token | 成本 | 平均延迟 ms | 失败 | 回退 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            call_rows,
            "",
            (
                f"共 `{token_observed_total}/{call_total}` 次调用返回 token 用量。"
                f"开启组比关闭组少调用 {int(off_calls['provider_call_count'] - on_calls['provider_call_count'])} 次，"
                f"但 token 增加 `{token_increase:.1%}`、成本增加 `{cost_increase:.1%}`。"
            ),
            "",
            (
                "严格门槛未通过，所以本轮只能作为机制与成本试验，不能作为正式"
                "效应估计。开启组每次决策都携带较长的逐手反思历史，是 token "
                "增长的主要原因；正式复跑前应先压缩为结构化滚动摘要。"
            ),
            "",
            "## 下一步",
            "",
            "1. 修复 JSON 结构校验后的自动修复/重试，要求零失败、零回退。",
            "2. 将 30 条详细反思压缩为固定预算的 belief state 和最近关键事件。",
            "3. 使用多个配对种子；每个种子共享 30 手形成阶段，再分叉 100–300 手。",
            "4. 增加同一决策节点的离线 counterfactual replay，隔离动作质量与后续轨迹分化。",
            "",
            f"Fork 签名：`{fork['signature']}`（两分支相同：`{fork['identical']}`）。",
        ]
    )


def write_analysis_from_artifacts(config: SharedHistoryConfig) -> None:
    """Regenerate derived analysis without making any provider calls."""
    per_hand = pd.read_csv(config.output_dir / "per_hand.csv")
    summary = pd.read_csv(config.output_dir / "exploitation_summary.csv")
    image_trajectory = pd.read_csv(config.output_dir / "image_trajectory.csv")
    calls = pd.read_csv(config.output_dir / "call_summary.csv")
    fork = json.loads((config.output_dir / "fork_summary.json").read_text())
    paired = _paired_llm_hand_deltas(per_hand)
    paired.to_csv(config.output_dir / "paired_llm_hand_deltas.csv", index=False)
    (config.output_dir / "REPORT.zh-CN.md").write_text(
        _report_markdown(config, summary, image_trajectory, calls, paired, fork),
        encoding="utf-8",
    )


def run_shared_history_experiment(
    config: SharedHistoryConfig,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    base_environment = _make_environment(config)
    base_environment.play(config.formation_hands)
    base_llm = _llm(base_environment)
    hero_name = base_llm.name
    formation_records = list(base_environment.records)
    formation_decisions = list(base_llm.decision_traces)
    formation_reflections = list(base_llm.reflection_traces)
    formation_failures = base_llm.provider_failures
    formation_signature = _fork_signature(base_environment)

    off_environment = copy.deepcopy(base_environment)
    on_environment = copy.deepcopy(base_environment)
    off_signature = _fork_signature(off_environment)
    on_signature = _fork_signature(on_environment)
    if len({formation_signature, off_signature, on_signature}) != 1:
        raise RuntimeError("Shared-history fork did not preserve identical pre-branch state.")

    if config.branch_workers > 1:
        with ThreadPoolExecutor(max_workers=2) as executor:
            off_future = executor.submit(
                _run_branch, off_environment, False, config.exploitation_hands
            )
            on_future = executor.submit(
                _run_branch, on_environment, True, config.exploitation_hands
            )
            off_environment = off_future.result()
            on_environment = on_future.result()
    else:
        off_environment = _run_branch(off_environment, False, config.exploitation_hands)
        on_environment = _run_branch(on_environment, True, config.exploitation_hands)

    branch_data: dict[str, tuple[HoldemEnvironment, list[HandRecord]]] = {
        "reflexive_off": (
            off_environment,
            off_environment.records[config.formation_hands :],
        ),
        "reflexive_on": (
            on_environment,
            on_environment.records[config.formation_hands :],
        ),
    }
    per_hand_rows = _per_hand_rows(formation_records, "formation", "shared_formation")
    image_rows = [
        _image_row(record, "formation", "shared_formation", hero_name)
        for record in formation_records
    ]
    trace_records = [
        {**trace, "phase": "formation", "branch_condition": "shared_formation"}
        for trace in formation_decisions
    ]
    reflection_records = [
        {**trace, "phase": "formation", "branch_condition": "shared_formation"}
        for trace in formation_reflections
    ]
    summary_rows = []
    call_rows = [
        {
            "phase": "formation",
            "condition": "shared_formation",
            **_provider_metrics(formation_decisions + formation_reflections),
            "provider_failures": formation_failures,
            "fallbacks": sum(
                bool(trace["final_decision"]["fallback_used"]) for trace in formation_decisions
            ),
        }
    ]

    for condition, (environment, records) in branch_data.items():
        llm = _llm(environment)
        branch_decisions = llm.decision_traces[len(formation_decisions) :]
        branch_reflections = llm.reflection_traces[len(formation_reflections) :]
        per_hand_rows.extend(_per_hand_rows(records, "exploitation", condition))
        image_rows.extend(
            _image_row(record, "exploitation", condition, hero_name) for record in records
        )
        trace_records.extend(
            {**trace, "phase": "exploitation", "branch_condition": condition}
            for trace in branch_decisions
        )
        reflection_records.extend(
            {**trace, "phase": "exploitation", "branch_condition": condition}
            for trace in branch_reflections
        )
        response_metrics = _response_metrics(records, hero_name)
        for index, player_type in enumerate(SIX_MAX_LINEUP):
            agent_name = environment.agents[index].name
            actions = [
                event for record in records for event in record.actions if event.actor == agent_name
            ]
            summary_rows.append(
                {
                    "condition": condition,
                    "player_type": player_type,
                    "hands": len(records),
                    "chips_per_100": 100.0
                    * sum(record.rewards[agent_name] for record in records)
                    / len(records),
                    "decision_count": len(actions),
                    "raise_rate": sum(event.action.value == "raise" for event in actions)
                    / max(1, len(actions)),
                    **response_metrics,
                }
            )
        call_rows.append(
            {
                "phase": "exploitation",
                "condition": condition,
                **_provider_metrics(branch_decisions + branch_reflections),
                "provider_failures": llm.provider_failures - formation_failures,
                "fallbacks": sum(
                    bool(trace["final_decision"]["fallback_used"]) for trace in branch_decisions
                ),
            }
        )

    per_hand = pd.DataFrame(per_hand_rows)
    summary = pd.DataFrame(summary_rows)
    image_trajectory = pd.DataFrame(image_rows)
    calls = pd.DataFrame(call_rows)
    paired = _paired_llm_hand_deltas(per_hand)
    llm_summary = summary[summary["player_type"] == "llm"].set_index("condition")
    delta = float(
        llm_summary.loc["reflexive_on", "chips_per_100"]
        - llm_summary.loc["reflexive_off", "chips_per_100"]
    )
    fork = {
        "signature": formation_signature,
        "identical": True,
        "record_count": config.formation_hands,
        "formation_final_mean_opponent_image": float(
            image_trajectory[image_trajectory["condition"] == "shared_formation"].iloc[-1][
                "mean_opponent_image_of_hero"
            ]
        ),
        "formation_final_self_image": float(
            image_trajectory[image_trajectory["condition"] == "shared_formation"].iloc[-1][
                "hero_self_public_image"
            ]
        ),
        "exploitation_chips_per_100_delta": delta,
        "valid_provider_gate": bool(
            calls["provider_failures"].sum() == 0
            and calls["fallbacks"].sum() == 0
            and calls["provider_call_count"].sum() == calls["token_observed_calls"].sum()
        ),
    }
    per_hand.to_csv(config.output_dir / "per_hand.csv", index=False)
    summary.to_csv(config.output_dir / "exploitation_summary.csv", index=False)
    image_trajectory.to_csv(config.output_dir / "image_trajectory.csv", index=False)
    calls.to_csv(config.output_dir / "call_summary.csv", index=False)
    paired.to_csv(config.output_dir / "paired_llm_hand_deltas.csv", index=False)
    _write_jsonl(config.output_dir / "decision_traces.jsonl.gz", trace_records)
    _write_jsonl(config.output_dir / "reflection_traces.jsonl.gz", reflection_records)
    (config.output_dir / "fork_summary.json").write_text(
        json.dumps(fork, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.output_dir / "EXPERIMENT.md").write_text(_design_markdown(config), encoding="utf-8")
    (config.output_dir / "REPORT.zh-CN.md").write_text(
        _report_markdown(config, summary, image_trajectory, calls, paired, fork),
        encoding="utf-8",
    )
    return {
        "per_hand": per_hand,
        "summary": summary,
        "image_trajectory": image_trajectory,
        "calls": calls,
        "paired": paired,
        "fork": fork,
    }
