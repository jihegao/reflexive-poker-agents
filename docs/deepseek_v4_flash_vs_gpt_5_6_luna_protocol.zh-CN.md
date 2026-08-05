# 《DeepSeek V4 Flash vs GPT‑5.6 Luna：只靠大脑、翻开 GTO、再带上自省和仿真，谁更会打德州？》

> 会解释 GTO，不等于能按 GTO 行动；会自省，也不等于能多赢筹码。

本文档定义一个三轮、配对、可审计的模型扑克比较协议。它是实验合同，不是现有结果的重新命名。任何未通过 provider gate、牌序镜像、身份校验、证据完整性和隐私检查的运行，都不得进入最终统计。

对应机器可读配置：

- `configs/deepseek_v4_flash_vs_gpt_5_6_luna.yaml`
- 校验命令：`PYTHONPATH=src python scripts/validate_showdown_protocol.py`
- 正式冻结检查：`PYTHONPATH=src python scripts/validate_showdown_protocol.py --formal`

当前配置的 `protocol.status` 是 `protocol_only`，GTO Reference Pack 也尚未冻结，因此 `--formal` 必须失败。这是预期的 fail-closed 行为。

## 1. 三个研究问题

1. 谁的原生扑克理解和策略更好？
2. 谁更能把外部知识转化为决策？
3. 谁更会利用自省和仿真，在多人博弈中形成优势？

这三个问题必须分轮回答。不能用一个混合总分把筹码收益、solver regret、成本、延迟和稳定性相加后称为“AI 智商”。

## 2. 统一比赛规则

- 无限注德州扑克现金局，初始筹码 `100 BB`，小盲 `0.5 BB`，大盲 `1 BB`，无抽水、无前注。
- 模型只返回结构化动作意图；PokerKit 负责发牌、合法动作、最小加注、全下、边池、摊牌和筹码守恒。
- 模型只看到本座位可见信息，不得看到对手底牌、对手私有工具结果、私有自省或隐藏审计日志。
- 下注尺寸离散为 `0.5 pot / 0.75 pot / 1 pot / all-in`。
- 相同牌序必须座位镜像，同一副牌交换模型位置再打一次；两个镜像分支状态完全隔离。
- 固定 prompt、采样参数、推理预算、历史窗口和最大输出长度。
- 模型身份对桌上 Agent 隐藏；公开回放可展示红蓝阵营。
- provider 超时、错误模型、fallback、版本漂移或未解决错误会使整个配对 block 无效。规则机器人不得代打。
- 不要求隐藏思维链，只记录短理由、置信度、动作意图和结构化工具状态。

## 3. 仓库现状与复用边界

### 可直接复用

- `llm_player.py` 的 provider adapter、结构化输出、模型身份、token、延迟和费用字段。
- Phase 1 的 fail-closed provider gate、调用前后 ledger、可恢复 block、source provenance 和不可变计划哈希。
- 已有的配对种子、bootstrap 区间、公开短理由和私有审计分层思想。
- 现有 OpenCode Go `deepseek-v4-flash` 与 Codex `gpt-5.6-luna` 的调用入口。

### 不能直接当作本测试证据

- 当前 `HoldemEnvironment` 是自研规则环境，不是 PokerKit 裁判。
- 当前六人实验是 `1 LLM + 5 heuristic agents`，不是 `3 DeepSeek + 3 Luna`。
- 当前 `LLMPlayer` 在 provider 失败或非法动作时会使用规则 fallback；本协议明确禁止这种接管。
- 当前 `reflexive_on/off` 把若干信息直接塞进上下文，不等同于显式、限额、可审计的 `reflect_update` 与 `equity_simulate` 工具。
- 当前 equity 估计是 Agent 内部预计算值，不是由模型提交对手范围后触发的统一 Monte Carlo 工具调用。
- 当前六人实验没有枚举全部 `C(6,3)=20` 阵营席位分配，也没有从相同历史 checkpoint 分叉三个影子条件。
- 当前输出文件名和公开/私有回放边界不满足本协议的完整证据合同。

因此，新测试应作为独立实验族实现，不应把旧 `second_order` 或 `phase1` 产物改名后复用。

## 4. 第一轮：裸脑单挑

Heads-up，纯 LLM，不开放工具、GTO 摘要、实时检索或 equity 计算结果。模型输入只包含自己的底牌、公共牌、底池、有效筹码、位置、合法动作、离散下注尺寸和固定长度公开行动历史。

样本：

- 热点首发：`50` 个牌序，双向座位镜像，共 `100` 手，仅标为探索性。
- 正式复核：至少 `500` 个牌序，双向座位镜像，共 `1,000` 手。
- 推断单位是牌序配对 block，不是单手独立样本。

指标：

- 配对净收益 `bb/100`
- 非法或自相矛盾动作率
- 相对离线 solver/reference policy 的决策 regret
- token、费用、端到端延迟和 provider 稳定性

## 5. 第二轮：冻结 GTO Reference Pack

仍是纯 LLM。每次调用附带同一版本、同一字节内容、同一哈希的静态参考包，不开放实时检索或 solver 工具。参考包至少覆盖 Heads-up 100BB 翻前范围、常用下注尺寸、牌面结构与频率原则，以及防守、诈唬和值下注的基本比例。

后台运行三条件：

- `raw`：无额外材料
- `masked_gto`：token 长度和版式匹配，但策略信息被遮蔽
- `gto`：真实冻结参考包

核心估计量：

\[
\Delta GTO = Regret_{Masked} - Regret_{GTO}
\]

并比较两个模型的差中之差。结论只能写成“参考冻结的 GTO 摘要后表现改善”，不能写成“模型实现了 GTO”。

正式运行前必须补齐 `reference_pack.path`、64 位 SHA-256，并把 `status` 改为 `frozen`。

## 6. 第三轮：六人阵营赛

主赛条件：

- 六人桌，`3 DeepSeek + 3 Luna`。
- Mesa 管理 Agent 生命周期、私有状态、历史和实验分叉。
- PokerKit 是唯一规则裁判。
- 枚举全部 `C(6,3)=20` 种阵营席位分配，并随机旋转按钮。
- 主赛开启 `reflect_update + equity_simulate`。

`reflect_update` 每手结束后最多调用一次，输出固定 schema 的私有状态：自己的策略漏洞、对手倾向、错误预测和下一手调整。它有固定 token 上限，不允许无限追加自由文本历史。

`equity_simulate` 每个决策最多调用一次，输入只能使用本座位可见信息，以及模型显式给出的对手范围。所有模型共享相同算法、rollout 数和随机种子规则；每次固定 `5,000` 次 Monte Carlo rollout。必须记录调用前动作意图、工具结果、调用后动作意图和是否翻转。

从相同公开历史和私有 checkpoint 分叉三个影子条件：

1. `tools_off`
2. `reflection_only`
3. `reflection_plus_simulation`

分叉后随机数流、模型调用和私有状态相互隔离。否则只能回答“带工具时谁赢了”，不能识别工具 uplift。

热点版：

\[
20\ \text{种席位} \times 2\ \text{个牌序种子} \times 20\ \text{手} = 800\ \text{手}
\]

正式版使用独立种子扩样，并在运行前冻结样本量。

主指标是两阵营总净收益。因恰好 `3 对 3` 且无抽水，阵营收益天然零和。辅助指标包括自省后动作改变率、同类错误减少、对手范围校准、仿真后动作翻转率、单位成本 regret 改善，以及工具失败、滥用和延迟。

## 7. 胜负判定

每轮报告配对 `bb/100` 和 95% 区间。区间跨过 `0` 时正式判为平局，不用样本均值强行制造赢家。

保留四块独立榜单：

| 榜单 | 回答的问题 |
|---|---|
| Return | 谁赢的筹码更多 |
| Understanding | 谁更接近正确规则、范围和策略 |
| Tool Uplift | 谁更能把 GTO、自省和仿真转化为行动 |
| Efficiency | 谁更快、更便宜、更稳定 |

节目叙事可以采用“三局两胜”，研究结论必须保留各指标的独立性。

## 8. 证据合同

每个可纳入统计的运行至少生成：

- `protocol.yaml`
- `model_manifest.json`
- `prompt_and_gto_hashes.json`
- `provider_gate.json`
- `hands.jsonl`
- `decisions.jsonl`
- `reflections.jsonl`
- `tool_calls.jsonl`
- `cost_ledger.csv`
- `summary.json`
- `public_replay.jsonl`
- `private_audit.jsonl`

`public_replay.jsonl` 必须剔除未摊牌底牌、私有工具结果和私有自省。`private_audit.jsonl` 仅进入隔离审计包，不能被下一座位或镜像分支读取。

## 9. 实现分层

建议新增独立的 `src/reflexive_poker/showdown/` 实验包，而不是继续扩张旧 `six_max_experiment.py`：

```text
src/reflexive_poker/showdown/
  protocol.py
  provider_gate.py
  pokerkit_engine.py
  mesa_arena.py
  agents.py
  tools.py
  pairing.py
  checkpoints.py
  regret.py
  evidence.py
  analysis.py
scripts/
  run_showdown.py
  analyze_showdown.py
```

实施 block：

- Block A：协议、模型身份 gate、hash manifest、paired-block 失效规则和公私日志 schema。
- Block B：PokerKit Heads-up runner、离散下注映射、双向镜像、分支隔离和筹码守恒测试。
- Block C：冻结 GTO/masked pack、预算匹配检查、离线 regret 评分和 `ΔGTO` 分析。
- Block D：Mesa 3v3、20 种席位、两个显式工具、相同 checkpoint 三分叉和阵营统计。
- Block E：provider smoke、探索性热点版、预注册正式版、公开回放和私有审计包。

## 10. 正式运行门槛

在以下条件全部满足前，不得启动确认性样本：

- `--formal` 协议校验通过
- PokerKit 与 Mesa 版本被冻结并记录
- 两个 provider 的小样本 smoke 为零未解决失败、零 fallback、身份完全匹配
- GTO pack 和 masked pack 均有固定哈希
- 镜像分支隔离测试通过
- 牌桌筹码守恒和边池回放测试通过
- solver/reference regret 评分版本被冻结
- 公共回放隐私测试通过
- Git worktree 干净，源代码和配置 fingerprint 已冻结
- 样本量和分析方法预注册

## 11. 发布节奏

先发布“第一夜观察”，清楚标注探索性结果和宽区间；后台继续执行预注册正式样本，再发布“复赛：第一夜的赢家经得起镜像和扩样吗？”。任何 pilot 都只证明链路能运行，不能证明模型稳定更强。
