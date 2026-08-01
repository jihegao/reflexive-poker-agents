from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .six_max_experiment import SixMaxConfig, run_six_max_experiment

MODEL_SPECS = (
    ("opencode-go", "deepseek-v4-flash"),
    ("codex", "gpt-5.6-luna"),
)
CONDITIONS = (
    ("reflexive_off", False),
    ("reflexive_on", True),
)


@dataclass(frozen=True)
class SecondOrderConfig:
    model_specs: tuple[tuple[str, str], ...] = MODEL_SPECS
    seeds: tuple[int, ...] = (9200,)
    hands: int = 6
    equity_samples: int = 64
    workers: int = 1
    bootstrap_samples: int = 5000
    output_dir: Path = Path("results/second_order/low_cost_pilot")


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(".", "_")


def _run_condition(
    config: SecondOrderConfig,
    provider: str,
    model: str,
    condition: str,
    reflexive_enabled: bool,
) -> dict[str, pd.DataFrame]:
    return run_six_max_experiment(
        SixMaxConfig(
            provider=provider,
            model=model,
            seeds=config.seeds,
            hands=config.hands,
            equity_samples=config.equity_samples,
            condition=condition,
            reflexive_enabled=reflexive_enabled,
            output_dir=config.output_dir / f"{_slug(provider)}__{_slug(model)}__{condition}",
        )
    )


def _bootstrap_interval(
    values: np.ndarray, samples: int, seed: int = 20260731
) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_rows(per_seed: pd.DataFrame) -> pd.DataFrame:
    llm = per_seed[per_seed["player_type"] == "llm"]
    off = llm[llm["condition"] == "reflexive_off"].copy()
    on = llm[llm["condition"] == "reflexive_on"].copy()
    paired = off.merge(on, on=["provider", "model", "seed"], suffixes=("_off", "_on"))
    paired["chips_per_100_delta"] = paired["chips_per_100_on"] - paired["chips_per_100_off"]
    paired["raise_rate_delta"] = paired["raise_rate_on"] - paired["raise_rate_off"]
    paired["decision_count_delta"] = paired["decision_count_on"] - paired["decision_count_off"]
    paired["token_delta"] = paired["total_tokens_on"] - paired["total_tokens_off"]
    paired["reported_cost_usd_delta"] = (
        paired["reported_cost_usd_on"] - paired["reported_cost_usd_off"]
    )
    return paired[
        [
            "provider",
            "model",
            "seed",
            "chips_per_100_off",
            "chips_per_100_on",
            "chips_per_100_delta",
            "raise_rate_off",
            "raise_rate_on",
            "raise_rate_delta",
            "decision_count_off",
            "decision_count_on",
            "decision_count_delta",
            "provider_call_count_off",
            "provider_call_count_on",
            "total_tokens_off",
            "total_tokens_on",
            "token_delta",
            "token_observed_call_count_off",
            "token_observed_call_count_on",
            "reported_cost_usd_off",
            "reported_cost_usd_on",
            "reported_cost_usd_delta",
            "cost_observed_call_count_off",
            "cost_observed_call_count_on",
            "mean_provider_latency_ms_off",
            "mean_provider_latency_ms_on",
            "latency_observed_call_count_off",
            "latency_observed_call_count_on",
            "provider_failure_count_off",
            "provider_failure_count_on",
            "fallback_count_off",
            "fallback_count_on",
        ]
    ]


def _paired_summary(paired: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows = []
    for (provider, model), group in paired.groupby(["provider", "model"]):
        values = group["chips_per_100_delta"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_interval(values, bootstrap_samples)
        total_calls = int(
            group["provider_call_count_off"].sum() + group["provider_call_count_on"].sum()
        )
        token_observed_calls = int(
            group["token_observed_call_count_off"].sum()
            + group["token_observed_call_count_on"].sum()
        )
        cost_observed_calls = int(
            group["cost_observed_call_count_off"].sum() + group["cost_observed_call_count_on"].sum()
        )
        latency_observed_calls = int(
            group["latency_observed_call_count_off"].sum()
            + group["latency_observed_call_count_on"].sum()
        )
        weighted_latency = (
            group["mean_provider_latency_ms_off"] * group["latency_observed_call_count_off"]
        ).sum() + (
            group["mean_provider_latency_ms_on"] * group["latency_observed_call_count_on"]
        ).sum()
        rows.append(
            {
                "provider": provider,
                "model": model,
                "paired_seeds": len(values),
                "mean_chips_per_100_delta": values.mean(),
                "median_chips_per_100_delta": float(np.median(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_seed_rate": float((values > 0).mean()),
                "mean_raise_rate_delta": group["raise_rate_delta"].mean(),
                "total_calls": total_calls,
                "total_tokens": (
                    int(group["total_tokens_off"].sum() + group["total_tokens_on"].sum())
                    if token_observed_calls == total_calls
                    else float("nan")
                ),
                "token_observed_calls": token_observed_calls,
                "reported_cost_usd": (
                    float(
                        group["reported_cost_usd_off"].sum() + group["reported_cost_usd_on"].sum()
                    )
                    if cost_observed_calls == total_calls
                    else float("nan")
                ),
                "cost_observed_calls": cost_observed_calls,
                "mean_provider_latency_ms": (
                    weighted_latency / latency_observed_calls
                    if latency_observed_calls
                    else float("nan")
                ),
                "provider_failures": int(
                    group["provider_failure_count_off"].sum()
                    + group["provider_failure_count_on"].sum()
                ),
                "fallbacks": int(
                    group["fallback_count_off"].sum() + group["fallback_count_on"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _design_markdown(config: SecondOrderConfig) -> str:
    models = ", ".join(f"{provider}/{model}" for provider, model in config.model_specs)
    return "\n".join(
        [
            "# Six-max second-order reasoning ablation",
            "",
            "## Claim",
            "",
            "Under identical deals, seats, five strategy opponents, numeric poker inputs and LLM family, compare first-order decisions with and without second-order table-image information.",
            "",
            "## Conditions",
            "",
            "- `reflexive_off`: cards, board, position, legal actions, stack, pot, multiway equity and pot odds only.",
            "- `reflexive_on`: adds own public image, opponent aggression/fold summaries, collective fold prediction and prior post-hand reflections.",
            f"- Models: {models}.",
            f"- Seeds: {', '.join(str(seed) for seed in config.seeds)}.",
            f"- Horizon: {config.hands} hands per condition and seed; 100 BB reset every hand.",
            f"- Equity Monte Carlo samples per decision: {config.equity_samples}.",
            "- Same seed is reused across both conditions, so deck order, seats and opponent initialization are paired.",
            "",
            "## Metrics and gate",
            "",
            "- Primary: paired LLM chips/100 difference, reflexive-on minus reflexive-off, analyzed separately by model.",
            "- Secondary: raise-rate difference, decision count, provider failures/fallbacks, tokens and latency.",
            "- Pilot success requires zero provider failures/fallbacks and complete paired artifacts. It does not validate profitability.",
            "- A confirmatory claim requires a preregistered multi-seed run whose paired 95% interval excludes zero and whose direction is not driven by one all-in pot.",
            "",
            "## Boundary",
            "",
            "The strategy opponents are compact heuristic agents, not solver-grade six-max baselines. Results test this simulator's second-order information channel, not real-world poker superiority.",
        ]
    )


def _report_markdown(paired_summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    lines = [
        "# 低成本 LLM 二阶推理配对实验报告",
        "",
        "本报告按模型分别比较 `reflexive_on - reflexive_off`。正值表示二阶信息条件在该模拟中获得更多筹码。",
        "",
        "| Provider/model | 配对种子 | chips/100 差 | 调用 | token | 报告成本 USD | 平均延迟 | 失败/回退 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired_summary.to_dict(orient="records"):
        tokens = "unknown" if pd.isna(row["total_tokens"]) else f"{int(row['total_tokens']):,}"
        cost = "n/a" if pd.isna(row["reported_cost_usd"]) else f"${row['reported_cost_usd']:.6f}"
        latency = (
            "n/a"
            if pd.isna(row["mean_provider_latency_ms"])
            else f"{row['mean_provider_latency_ms'] / 1000:.2f}s"
        )
        lines.append(
            f"| {row['provider']}/{row['model']} | {row['paired_seeds']} | "
            f"{row['mean_chips_per_100_delta']:+.2f} | {row['total_calls']} | {tokens} | "
            f"{cost} | {latency} | {row['provider_failures']}/{row['fallbacks']} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是低成本 pilot。若每个模型只有一个配对种子，结果没有可估计的跨种子不确定性，不能确认二阶策略有效；它只验证干预、配对和真实 provider 链路。",
            "",
            f"保存的配对行数：{len(paired)}。后续确认性实验应扩大配对种子，并审计收益是否由单个全下底池主导。",
        ]
    )
    return "\n".join(lines)


def run_second_order_experiment(config: SecondOrderConfig) -> dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (provider, model, condition, enabled)
        for provider, model in config.model_specs
        for condition, enabled in CONDITIONS
    ]
    if config.workers > 1:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            results = list(
                executor.map(
                    lambda job: _run_condition(config, job[0], job[1], job[2], job[3]),
                    jobs,
                )
            )
    else:
        results = [_run_condition(config, *job) for job in jobs]
    per_seed = pd.concat([result["per_seed"] for result in results], ignore_index=True)
    all_agents_summary = pd.concat([result["summary"] for result in results], ignore_index=True)
    paired = _paired_rows(per_seed)
    paired_summary = _paired_summary(paired, config.bootstrap_samples)
    per_seed.to_csv(config.output_dir / "per_seed_all_agents.csv", index=False)
    all_agents_summary.to_csv(config.output_dir / "all_agents_summary.csv", index=False)
    paired.to_csv(config.output_dir / "paired_llm.csv", index=False)
    paired_summary.to_csv(config.output_dir / "paired_summary.csv", index=False)
    (config.output_dir / "EXPERIMENT.md").write_text(_design_markdown(config), encoding="utf-8")
    (config.output_dir / "REPORT.zh-CN.md").write_text(
        _report_markdown(paired_summary, paired), encoding="utf-8"
    )
    return {
        "per_seed": per_seed,
        "all_agents_summary": all_agents_summary,
        "paired": paired,
        "paired_summary": paired_summary,
    }
