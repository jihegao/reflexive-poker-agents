# PRBench Phase 1 停止记录与后续计划

## 记录结论

本记录对应正式 run `20260803T124112Z-2366fa6f8a`（tag：`prbench-phase1-refrozen-v6`）。该 run 按用户指示于 **2026-08-03 13:22:44 UTC（21:22:44 Asia/Shanghai）** 通过 `expctl run stop` 停止，终态为 `cancelled`、exit code `143`。

实验没有完成，因此本 run **不是有效的 Phase 1 论文证据**，不得生成或宣称论文 outcome。所有已产生的 raw attempts、predictions、ledger、provider gate、source provenance 和事件日志均保留为审计材料。

权威记录：

- run 元数据：`results/experiments/20260803T124112Z-2366fa6f8a/run.json`
- 生命周期：`results/experiments/20260803T124112Z-2366fa6f8a/events.jsonl`
- 冻结 provenance：`results/experiments/20260803T124112Z-2366fa6f8a/artifacts/SOURCE_PROVENANCE.json`
- 价格快照：`results/experiments/20260803T124112Z-2366fa6f8a/frozen_inputs/PRICE_MANIFEST.json`

## 正式计划基线

计划文件为 [`prbench_cross_model_experiment_plan.zh-CN.md`](prbench_cross_model_experiment_plan.zh-CN.md)，执行配置为 [`configs/phase1.yaml`](../configs/phase1.yaml)。Phase 1 的最小证据包要求：

1. DeepSeek-V4-Flash + OpenCode Go、GPT-5.6-Luna + Codex 两个 serving systems。
2. 每个模型 200 cases × 5 treatments（共 1000 个 primary offline predictions）。
3. 两模型 provider preflight 各 20 predictions，覆盖五个 treatment。
4. Heads-up 闭环覆盖 fixed/adaptive 两个 regime、三个 treatments、30 个 paired seeds，并通过完整 paired-block、失败、成本和 provenance 门禁。
5. 保存 raw attempts、ledger、成本、逐 case 评分、轨迹级推断、paired inference、Holm 校正和中文报告；regret 是主指标，return 是次级指标。
6. Phase 1 outcome lock 之前不启动 Phase 2 outcome，也不把 preflight、mock 或不完整 run 写成论文结论。

## 本次 run 的过程与结果

### 冻结与预检

- config hash：`a43165384140483bc96d88cd62f3756a9e0369da3cfdd71cbacf12369baf82e7`
- source fingerprint：`fc80151441eb8c48017bd739b899267213d596b618b777d960c622b45e694b82`
- source snapshot SHA-256：`b63e85ab99fa15dfa4ef074ca9d18c0938c9ecf48cefbbcd4c6c783eedc87c74`
- git commit：`fcb3a2c52962c557a833c8b2a637ed206bf0342e`
- worktree：`dirty=false`
- price manifest SHA-256：`da9c708135d10940236f476eac4603c347244afabee4173f4ef653fae41ea6a1`
- protocol：`prbench-cross-model-v1`

provider preflight 已完成且两个 gate 均为 valid：

| serving system | expected | observed | gate | raw failures | retries |
|---|---:|---:|---|---:|---:|
| Codex / GPT-5.6-Luna | 20 | 20 | valid | 0 | 0 |
| OpenCode Go / DeepSeek-V4-Flash | 20 | 20 | valid | 0 | 0 |

生命周期事件为 `run_created → pricing_frozen → run_started → source_frozen → provider_preflight completed → offline_understanding started → run_cancelled`。

### 停止时的离线进度

DeepSeek 正式离线阶段在停止时仍是未完成的 live artifact：

- completed predictions：`47 / 1000`
- `live_predictions.jsonl`：47 行
- `live_provider_attempts.jsonl`：48 行
- ledger calls：49
- retries：1
- raw failures：1
- unresolved failures：0
- token/cost observed calls：48

其中唯一 raw failure 为 call `#27`（primary，`retry=false`）：

```text
ValueError: type_probabilities must sum to one
```

schema repair call `#28`（`retry=true`）成功。原始失败没有被删除；它仍然是审计记录。由于正式离线未达到 1000 predictions，且 Codex 正式离线、Heads-up 闭环均未开始，该 run 不能通过 Phase 1 completion gate。

停止时 `LIVE_PROGRESS.json` 仍可能保留 `state=running`，这是被 SIGTERM 中断的 live worker 文件状态；run-level 的 `run.json` 与 `events.jsonl` 已记录 `cancelled`/`run_cancelled`，以 run-level 终态为权威。

## 明确未完成项

| 要求 | 当前证据 | 判断 |
|---|---|---|
| 双模型各 1000 个 offline predictions | DeepSeek 47/1000；Codex formal artifact 不存在 | 未完成 |
| 双模型正式 offline gate | 未产生终态 gate | 未完成 |
| Heads-up paired closed-loop | 未开始；无 `CROSS_MODEL_PAIRED_BLOCK_STATUS.json` | 未完成 |
| Phase 1 evidence bundle | 未生成 `PHASE1_EVIDENCE_STATUS.json` | 未完成 |
| 论文 outcome / model ranking | 没有合法 outcome 输入 | 禁止宣称 |
| Phase 2 四模型 outcome | 未启动 | 按计划禁止启动 |

## Subagent 过程记录

- `statistics_audit` 发现 fixed-regime checkpoint 过滤和 large-pot paired trim 两个 P0 统计缺口；补丁经测试后合入主分支，形成 `fcb3a2c`，并据此重新冻结 v6。
- `phase2_gap_audit` 在独立 worktree 完成 Phase 2 evidence admission hardening，提交 `3b3242d`；由于 Phase 1 尚未锁定，**尚未合入 main**。
- `phase2_runner` 对 v6 做只读生命周期、attempt/ledger、provider failure 和 provenance 交叉监控；在 call #27 failure 及 #28 repair 时保留并报告了审计信息。
- `provider_provenance` 和其他隔离 subagent 完成 provider identity、价格和成本可观测性验证；未对本次未完成 run 伪造完成状态。

## 后续执行计划

### A. 停止后的证据封存（已完成）

1. 确认 run 终态为 `cancelled/143`。
2. 保留 v6 的 source snapshot、price manifest、raw attempts、live predictions、ledger 和 events。
3. 将 v6 标记为 audit-only；不把 partial artifacts 混入任何新正式 run。

### B. Phase 1 重新启动前的审查（待批准）

1. 复核单 worker（`max_workers: 1`）的运行时间和预算；不要在冻结 run 中临时改并发或 treatment。
2. 决定采用同一精确命令 resume，还是在新的 run id 下重新冻结；任何新 run 都必须重新记录 config、source、price、prompt/schema 和 case-manifest hash。
3. 保持 provider gate：每个决策最多一次 schema repair、zero unresolved failures、zero fallback、exact identity、完整 token/cost accounting。

### C. Phase 1 正式证据（重启后）

1. 完成双模型各 1000 个 offline predictions，并检查 raw attempts 与 ledger 对账。
2. 完成 30 paired seeds 的 Heads-up fixed/adaptive 闭环；不完整 paired block 不进入分析。
3. 运行 `audit_phase1_evidence_bundle(...)`，生成 `PHASE1_EVIDENCE_STATUS.json` 与 `PHASE1_EVIDENCE_AUDIT.zh-CN.md`。
4. 只有当 `complete=true`、provenance 一致、worktree clean、价格 hash 一致且所有失败门禁通过时，才锁定 Phase 1 evidence bundle。

### D. Phase 2 四模型扩展（仅在 Phase 1 lock 后）

1. 合入已隔离验证的 `3b3242d`，重新运行测试和 diff 检查。
2. 运行 `paper-phase2-preflight`，覆盖 DeepSeek、Qwen、GLM、GPT-5.6-Luna 的 identity、schema、token/cost 和 provider gate。
3. 固化 Qwen/GLM serving-system 版本、价格清单和 run-local 输入快照。
4. 以 Phase 1 evidence lock 绑定功效分析，再运行 fail-closed 的四模型 offline 与 Six-max 外部有效性扩展。
5. 单独报告 Understanding、Regret、Return、Cost、Latency 和 Valid blocks；不得把不同 serving stack 混写成单一“模型能力”。

## 当前提交边界

本记录只新增审计/计划文档，不修改冻结源码、配置或实验 artifacts。提交前必须确认：

```bash
git status --short --branch
git diff --check
```

本文件与 v6 artifacts 一起构成“已停止、未完成、可审计”的过程记录；它不是 Phase 1 结果报告，也不是 Phase 2 完成证明。
