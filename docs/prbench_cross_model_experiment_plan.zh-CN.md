# PRBench 扑克递归推理两阶段实验计划

状态：Phase 1 基础设施已实现，待双 provider 正式预检后冻结预注册版本
协议代号：`prbench-cross-model-v1`
日期：2026-08-02

## 1. 研究目标

本实验在统一扑克环境、输入信息、动作空间和审计协议下分两阶段比较 serving systems。

Phase 1 论文最小证据包固定为：

- `opencode-go/deepseek-v4-flash`
- `codex/gpt-5.6-luna`

Phase 2 扩展候选为：

- `qwen3.7`，正式运行前冻结具体变体和版本 ID
- `glm-5.2`

核心问题不是“哪个模型偶然赢得最多”，而是：

1. 哪个模型对牌桌状态和对手状态理解得最准？
2. D2/D3 递归信息是否比 D0/D1 带来额外理解增益？
3. 更准确的理解能否降低决策 regret，并转化为长期收益？
4. 收益提升是否足以覆盖额外 token、费用和延迟？

主排行榜同时展示 `Understanding ↑`、`Cost ↓`、`Return ↑`，不预设跨单位加权总分。综合比较使用 Pareto 支配关系。

## 2. 证据边界

本实验衡量的是模型在本仓库离散化德州扑克仿真器中的战略、递归和适应推理能力。它不是竞技扑克 solver benchmark，也不能直接证明模型在真实牌桌、其他博弈或一般社会推理中的能力。

若四个模型通过不同 CLI、网关或 agent runtime 调用，结果必须表述为“模型 + serving stack”的部署系统比较。只有在系统提示、上下文处理、重试策略和结构化输出协议足够一致时，才可进一步讨论模型层差异。

模型不需要输出隐藏思维链。实验只保存可验证的结构化判断、简短审计摘要、最终动作和 provider 元数据。

## 3. 预注册假设

- H1：Phase 1 两个 serving systems 的牌桌理解和对手理解准确性存在可重复差异。
- H2：在切换型、适应型对手上，`D2/D3` 的对手理解优于 `D0/D1`。
- H3：对手行动分布预测越准确，后续动作 regret 越低。
- H4：理解提升能否转化为收益取决于 opponent regime，不能由静态理解分数直接推出。
- H5：成本、准确性和收益之间形成效率前沿，不预期存在所有维度上的必然冠军。

## 4. 实验因子

| 因子 | 条件 |
| --- | --- |
| Model | Phase 1 两个 serving systems；Phase 2 扩为四个 |
| Reasoning treatment | D0、D1、D1-BM、D2、D3 |
| Opponent type | Rock、TAG、LAG、Calling Station、Myopic |
| Opponent regime | 固定、带噪声、中途切换、持续适应 |
| Arena | Heads-up 主实验；Six-max 外部有效性验证 |
| Repetition | 相同 seed、牌序、座位、对手随机流和形成期 checkpoint |

推理层级冻结为：

- D0 `state_only`：仅当前公开牌桌状态。
- D1 `action_prediction`：增加历史统计并预测对手下一行动。
- D1-BM `d1_budget_matched`：保留 D1 实质信息，以中性长度控制匹配递归输入结构和输出预算。
- D2 `recursive_d2`：增加“对手如何看待 Hero”。
- D3 `recursive_d3`：进一步预测对手调整及 Hero 对该调整的反向利用。

`history_statistics` 和 `strategy_type` 保留为规则仿真中的机制拆分条件，不进入第一版真实模型主排行榜，除非 pilot 表明它们是解释 D1/D2 差异所必需的对照。

## 5. 统一决策输出

每次真实模型决策必须返回严格 JSON，至少包含：

```json
{
  "table_state": {
    "street": "turn",
    "position": "button",
    "pot_bb": 18.0,
    "effective_stack_bb": 74.0,
    "spr": 4.11,
    "pot_odds": 0.25,
    "hand_class": "top_pair",
    "equity": 0.63,
    "legal_actions": ["fold", "check_call", "raise"]
  },
  "opponent_state": {
    "type_probabilities": {
      "rock": 0.05,
      "tag": 0.20,
      "lag": 0.60,
      "calling_station": 0.10,
      "myopic": 0.05
    },
    "action_probabilities": {
      "fold": 0.20,
      "check_call": 0.30,
      "raise": 0.50
    },
    "hero_image_aggression": 0.80,
    "adaptation_probability": 0.70
  },
  "action": "check_call",
  "raise_scale": 0.0,
  "confidence": 0.74,
  "audit_summary": "short non-CoT summary"
}
```

概率字段必须在 `[0, 1]`，每个概率分布之和必须在数值容差内等于 1。Schema repair 最多允许一次；未解决失败使整个 paired seed block 失效。

## 6. Ground truth 与理解评分

### 6.1 牌桌理解

| 子项 | Ground truth | 指标 |
| --- | --- | --- |
| street、position | 环境状态 | Exact Match |
| pot、stack、SPR、pot odds | 环境计算 | 归一化绝对误差 |
| legal actions | 规则引擎 | Set Accuracy |
| hand class | 牌型求值器 | Accuracy |
| equity | 对真实生成策略范围的精确枚举或高精度模拟 | MAE、Brier |

Equity 不按对手本局实际暗牌评分。模型无法观察该暗牌；有效 ground truth 应是基于环境真实条件范围和策略分布的期望 equity。

分别报告基础状态读取和实质推理分数，防止 street、pot 等容易字段掩盖 equity 误差：

- `U_table_basic`
- `U_table_hand`
- `U_table_equity`
- `U_table_total`

### 6.2 对手理解

| 子项 | Ground truth | 指标 |
| --- | --- | --- |
| 潜在策略类型 | 环境初始化的真实类型 | Multiclass Brier、Log Loss、Top-1 Accuracy |
| 下一行动分布 | 对手在当前信息集的真实策略分布 | Brier、Cross-entropy |
| Hero image | 对手内部实际追踪状态 | MAE |
| 策略切换 | 环境真实切换点 | 检测率、Detection Delay |
| 适应方向 | 对手真实内部调整状态 | MAE、方向准确率 |

这里的“对手心理 ground truth”只指模拟对手内部可读取的策略变量，不把不可验证的自然语言心理解释当作真值。

主报告保留 `U_table` 和 `U_opponent` 两列。可以输出预注册权重的 `U_total`，但不能只发布总分。

## 7. 成本度量

每次调用保存：

- 声明 provider/model 与 provider 返回的实际模型/版本 ID；
- input、output、reasoning、cached tokens；
- 实际计费、延迟、request/response ID；
- retry、schema failure、provider failure、fallback；
- prompt hash、schema hash、history window 和调用参数。

主要成本指标：

- `USD / 100 valid decisions`
- `tokens / valid decision`
- `latency p50 / p95`
- `USD / 100 hands`
- `U_opponent / USD`

`chips / 1000 tokens` 仅作辅助指标，因为收益接近零或为负时该比率不稳定。

订阅或本地 CLI 没有逐次账单时，不得记录为零成本。分别保存：

- `observed_billed_cost`
- `estimated_api_equivalent_cost`
- `cost_observability = exact | estimated | unavailable`

正式运行前冻结带日期的价格清单；实验完成后不得用新价格回写旧运行的实际成本。

## 8. 收益度量

主收益指标为形成期结束后的：

```text
post_formation_chips_per_100
```

每个 seed 先运行完全相同的 formation，然后从相同 checkpoint fork。所有模型和 treatment 必须共享：

- 牌序和公共牌；
- 座位及座位镜像；
- 对手初始化及随机数流；
- 形成期状态；
- 策略切换点。

每个模型报告：

- chips/100；
- 配对差值及 95% CI；
- 正收益 seed 比例；
- 最差四分位收益；
- seed 间标准差；
- 最大单手贡献；
- 去除绝对收益最高 1% 手牌后的结果；
- leave-one-large-pot-out 结果。

若环境可以稳定估计动作 EV，增加 `Decision Regret = EV(a*) - EV(a_model)` 作为降方差机制指标，但不替代真实收益。

## 9. 执行阶段：论文最小证据包优先

旧版 `Stage 0–4` 把 provider smoke、规则筛选和真实模型证据分散到多个阶段，容易出现“工程链路已经跑通，但第一阶段仍不能支撑论文结论”的状态。现将执行协议重组为两个研究阶段：

- **Phase 1：两模型论文最小证据包**。必须产出理解、等预算对照、传统概率基线、闭环 regret、收益稳健性和成本证据。
- **Phase 2：四模型与 Six-max 扩展**。在 Phase 1 结论成立后，加入 Qwen、GLM、异质 Six-max 和更大收益样本。

### Phase 1.0：冻结协议与双模型预检

第一阶段固定两个已有真实调用路径：

- `opencode-go/deepseek-v4-flash`
- `codex/gpt-5.6-luna`

每个 serving system 在 D0、D1、D1-BM、D2、D3 上各完成 4 个结构化 case，共 20 次预检调用。要求：

- 严格概率 JSON Schema 通过，两个概率分布分别归一化；
- 返回 provider/model identity 与 manifest 一致；
- zero unresolved provider failure；
- zero fallback；
- token、latency 和可观测成本字段完整；
- 在任何 outcome 调用前冻结带日期的价格快照；快照的 SHA-256、原始字节与冻结时间必须进入 run metadata、artifact 和源码归档；
- prompt、history window、action space、输出上限和 temperature 冻结；
- 原始失败和 schema repair 尝试进入独立审计账本。

预检只决定能否开始正式实验，不进入论文结果。任一 serving system 门禁失败时，必须先修复 adapter，不能用 mock 或另一模型静默替代。

### Phase 1.1：200-case 离线理解正式实验

离线数据集由 50 条独立轨迹各抽取 4 个冻结 checkpoint：

```text
5 opponent types × 2 regimes × 5 seeds × 4 checkpoints = 200 cases
```

两个 regime 为 `fixed` 与 `adaptive_shift`；checkpoint 覆盖短历史、切换前长历史、切换后立即和切换后稳定期。preflop、flop、turn、river 通过确定性轮换保持平衡。独立重复单位是轨迹，不是单个 case。

每个真实模型运行：

- D0 `state_only`；
- D1 `action_prediction`；
- D1-BM `d1_budget_matched`：不含递归信息，但结构、输出 schema 和输入长度控制匹配递归条件；
- D2 `recursive_d2`；
- D3 `recursive_d3`。

每模型 1,000 次正式调用。另行预注册 20 个重复 case 用于稳定性检查时，必须在 manifest 中冻结 case ID，且重复结果不混入主样本量。

同一数据集还运行无需模型调用的：

- Oracle；
- Uniform；
- Laplace-smoothed Frequency；
- 固定类型 Bayesian filter；
- 带冻结转移概率的 HMM filter。

主检验为动态对手切换后的 `D2 - D1-BM` action/type Brier 差异，以及该差异相对 fixed regime 的交互。D3 相对 D2 是次要层级检验，不能通过观察结果后“择优”进入主闭环。

### Phase 1.2：规则与指标 sanity check

规则代理不再承担选择 D2/D3 的职责，只用于验证：

- Oracle、Bayesian/HMM、Frequency、Uniform 的指标方向；
- shared-formation fork hash；
- D1-BM 确实屏蔽递归字段；
- counterfactual action utility 与 regret 单位；
- top-1% large-pot trim 和 paired statistics；
- 可恢复 block、原子完成标记和预算账本。

40-cell、60-seed 的完整规则矩阵可继续作为便宜的模拟敏感性分析，但不是 Phase 1 完成门槛，也不能替代真实模型论文证据。

### Phase 1.3：双模型 Heads-up 闭环正式确认

真实模型闭环只运行预注册的三个 treatment：

- D0 `state_only`；
- D1-BM `d1_budget_matched`；
- D2 `recursive_d2`。

主矩阵为：

```text
2 serving systems × 3 treatments × 2 regimes × 40 paired seeds
```

若冻结的调用率估计表明 40 seeds 超出预算，可以在查看任何正式 outcome 前统一降到 30；低于 30 不作为论文正式确认。每个 block 使用共享 formation checkpoint、相同牌序、对手随机流、切换状态和座位镜像。一个 arm 失败则整个 paired block 退出主分析。

闭环主指标是 counterfactual decision regret；`post_formation_chips_per_100` 为重要次要指标。只有收益 CI、座位镜像、top-1% trim 和 leave-largest-pot-out 全部同向时，才允许写“improves return”。否则结论限定为 opponent-state estimation 和 decision quality。

每模型总上限保持 10,000 calls：1,600 留给离线理解与重复，8,000 留给 fixed/adaptive Heads-up，400 留给预检与 schema repair。运行中不得依据中间收益提前停止或重新分配条件预算。

### Phase 1.4：锁定分析与论文证据包

Phase 1 完成必须同时具备：

- 冻结 manifest、源码指纹、prompt/schema hash 和价格快照；
- 200-case raw cases、raw predictions 和逐 case scores；
- 两模型分别报告的 Brier、校准、regret、成本与 provider gate；
- Heads-up per-hand、per-seed、paired、large-pot sensitivity 和失败尝试；
- 预注册 contrast、cluster bootstrap、paired permutation 和 Holm 校正；
- 自动生成的中文报告与机器可读分析摘要。

若一个模型成立、另一个不成立，结论必须写为 model-dependent；若 regret 成立但收益不成立，结论不得升级为长期盈利优势。

### Phase 2：四模型和外部有效性扩展

Phase 2 才加入：

- 冻结版本的 Qwen 与 GLM serving system；
- 四模型完整 Pareto 排行；
- 异质 Six-max 外部有效性；
- 更长 horizon、更大 paired seed 数和收益功效分析；
- 需要时追加 D3 正式闭环，而不是根据 Phase 1 正文结果临时挑选。

冻结的准备清单见 [`configs/phase2.yaml`](../configs/phase2.yaml)：它登记四个
serving systems、全部五个理解处理条件，以及异质 Six-max 外部有效性不变量。
`expctl paper-phase2-preflight` 只执行这四个系统的四案例、全处理条件 provider
gate；它不运行任何 Phase 2 outcome，因此不能把这次预检误作完整横评或已接入的
Six-max 证据。

### Agent-friendly CLI 与前端隔离

研究实验不进入前端请求链路。`expctl` 只提交独立后台 worker，并将状态、事件、检查点和结果写入独立 registry：

```bash
expctl doctor --output json
expctl experiment list --output json
expctl config validate configs/phase1.yaml --output json

expctl run start \
  --config configs/phase2.yaml \
  --experiment paper-phase2-preflight \
  --request-id phase2-provider-preflight-v1 \
  --tag phase2-provider-preflight \
  --output json

expctl run start \
  --config configs/phase1.yaml \
  --experiment paper-phase1 \
  --request-id phase1-formal-v1 \
  --tag paper-phase1 \
  --output json

expctl run status <run-id> --output json
expctl run logs <run-id> --follow --format jsonl
expctl run pause <run-id> --output json
expctl run resume <run-id> --output json
expctl run stop <run-id> --output json
expctl analyze <run-id> --output json
expctl export <run-id> --format tar.gz --output json
```

控制协议保证：

- `start` 通过 `--request-id` 幂等；
- worker 脱离终端运行，终端断开不终止实验；
- 状态固定为 `created → queued → running → completed`，并支持 `paused`、`failed`、`cancelled`；
- JSON 错误包含稳定的 `code`、`retryable` 和 `details`；
- 默认单 worker、单数值计算线程并降低进程优先级；
- 实验目录不访问前端数据库，也不自动发布 prompt、策略或模型配置；
- `resume` 复用 Phase 1 的 cell/seed/block checkpoints，不把不完整 block 当正式证据。

## 10. 统计分析

独立重复单位是 `seed/fork block`，不是单手或单次决策。

- 理解分数：按 seed/episode 聚类 bootstrap。
- 收益比较：paired bootstrap 和 paired permutation test。
- 多重比较：Holm 校正。
- 主交互：`Model × D-level × Opponent regime`。
- 概率校准：Brier decomposition、reliability curve、ECE。
- 稳健性：top-1% large-pot trim、leave-largest-pot-out、worst quartile。
- 机制分析：`Understanding → Decision Regret → Return`，标记为探索性 mediation，不作未经随机化支持的因果结论。

失败调用不进行结果插补。paired block 只有在所有对应条件的模型调用、成本记录和 fork 验证均完整时才进入主分析。失败和重跑尝试单独报告。

## 11. 有效性门禁

一个正式 block 仅在以下条件全部满足时有效：

1. source fingerprint、protocol hash 和 prompt/schema hash 匹配冻结 manifest；
2. shared formation fork hash 一致；
3. exact provider/model identity 匹配；
4. zero unresolved failures；
5. zero fallback；
6. paired arms 调用完整且平衡；
7. token、cost、latency accounting 完整或明确标记不可观测；
8. 所需原始 trace 和 seed-level rows 均已原子写入。

任何门禁失败时，不得以“最终动作碰巧相同”为理由保留该 block。

## 12. 产物结构

建议保存到：

```text
results/prbench_cross_model/<run_id>/
  manifest.json
  preregistration.md
  provider_preflight.csv
  pricing_manifest.json
  offline_understanding/
    cases.jsonl.gz
    predictions.jsonl.gz
    scores_per_case.csv
    scores_per_model.csv
  closed_loop/
    per_hand.csv
    per_seed.csv
    paired.csv
    paired_hand_deltas.csv
    decision_traces.jsonl.gz
    provider_ledger.json
    provider_gate.json
  analysis/
    understanding_summary.csv
    cost_summary.csv
    return_summary.csv
    inference.csv
    pareto_frontier.csv
    REPORT.zh-CN.md
```

原始行级数据必须先保存，再生成汇总和图表。截图不能作为唯一证据。

## 13. 结果表与结论规则

Phase 1 主证据表不强行形成总排行榜：

| Serving system | D2-D1BM Action Brier ↓ | Type Brier ↓ | Regret reduction ↑ | Cost/100 decisions ↓ | chips/100 Δ | Valid blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash + OpenCode Go | | | | | | |
| GPT-5.6-Luna + Codex | | | | | | |

Phase 2 四模型扩展榜：

| Model | Table U ↑ | Opponent U ↑ | Cost/100 decisions ↓ | Latency p95 ↓ | chips/100 ↑ | Valid blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash | | | | | | |
| Qwen3.7 | | | | | | |
| GLM-5.2 | | | | | | |
| GPT-5.6-Luna | | | | | | |

模型 A Pareto 支配模型 B，当且仅当：

```text
U_A >= U_B
R_A >= R_B
C_A <= C_B
```

且至少一项严格更优。若置信区间大量重叠，结论写为“当前预算下未能区分”，而不是强行排列名次。

## 14. 实现与正式运行检查单

已实现的 Phase 1 基础设施：

- [x] Ground-truth case exporter 与模型可见字段分离。
- [x] 严格 opponent/type/action 概率 Schema 和归一化验证。
- [x] D1-BM 输入屏蔽与长度控制。
- [x] Oracle、Uniform、Frequency、Bayesian、HMM 基线。
- [x] 离线 200-case 确定性生成、raw prediction 和逐 case 评分。
- [x] 精确抽象 action utility 与 counterfactual regret。
- [x] shared fork、provider ledger、失败门禁和 resumable block。
- [x] 后台 `expctl`、JSON/JSONL、幂等 start、pause/resume/stop、analyze/export。

开始正式收费实验前仍需完成：

- [ ] DeepSeek 与 Codex 各 20 次、覆盖五个 treatment 的真实 provider 预检。
- [x] 价格快照的 hash-lock、run-local artifact 和证据审计门禁已实现；每次正式 run 仍需在调用前验证它。
- [ ] 冻结模型版本、价格、prompt、schema、case manifest 和源码指纹。
- [ ] 在独立干净 worktree 执行双模型 200-case 正式离线实验。
- [ ] 在查看 outcome 前冻结 30 或 40 paired seeds 和闭环预算。
- [ ] 完成双模型 Heads-up 正式闭环及全部有效性门禁。
- [ ] 完整 raw artifacts、失败尝试、成本和报告归档。
- [ ] 报告明确区分 baseline、mock、preflight、valid live-model evidence。

Phase 2 扩展项：

- [ ] 冻结并接入 Qwen 与 GLM 的具体 serving system 版本。
- [ ] 四模型统一预检和正式 Pareto 横向比较。
- [ ] Six-max 外部有效性与更高功效的收益实验。
