# Phase 1 无效闭环运行审计（2026-08-02）

这不是 Phase 1 结果，也不能用于任何收益、regret、模型比较或论文表格。

## 已排除的运行

- 正式运行 ID：`20260802T203159Z-36b4e2c73b`
- 配置：`configs/phase1.yaml`，`opencode-go/deepseek-v4-flash`，fixed Heads-up，seed `9700`
- 运行状态：操作者在发现配置必然无效后终止；registry 记录为 `failed / WORKER_DISAPPEARED`。
- 唯一完成 block：`seed_9700`，`provider_gate.valid=false`，不可进入任何主分析。

该 block 的审计账本为 106 次总调用、100 次主调用、6 次重试、7 次原始失败和 1 次未解决失败。它还记录了 44 次 fallback，且三 treatment 调用不平衡；因此不满足 zero-unresolved-failure、zero-fallback、exact serving identity source 和 balanced paired arms 门禁。

## 根因与处置

冻结配置为一个 20 手、5 手 formation、三个 treatment 的 paired block 分配了 100 次主调用。实现中的合法动作上界是 600 次；独立的 mock 校准 block 也实际使用了 159 次。因此，100 次 cap 会把剩余决策变成 fallback，并使 block 必然失效。

原始 provider 失败均为概率向量的和不为 1（`action_probabilities` 或 `type_probabilities`）；旧 retry 请求不携带校正反馈。修复后，重试会明确说明先前验证错误并要求两个概率分布精确归一；`ProviderBudgetExceeded` 会向上 fail-closed，而不再被转换为规则策略 fallback。

## 防复发门禁

后续 `llm-confirmation` 在任何 provider preflight 前强制验证：

1. 每 paired block 主调用 cap 至少覆盖所有合法动作的上界；
2. 每个 fixed/adaptive job 的主调用配额至少覆盖所有已冻结 seeds；
3. `max_calls_per_model`、offline、retry reserve 和 Heads-up 配额由 YAML 显式传递，不再在 runner 内静默硬编码。

对原 YAML 的无收费启动前检查运行 `20260802T212345Z-1527cd4f8d` 已在 preflight 之前以 `cap=100, required_upper_bound=600` 失败，未创建 provider 调用 artifact。

在新的成本授权和重新冻结的样本量/预算获得确认前，不得重新启动正式 outcome 调用。此前的离线双模型 evidence 及其审计 artifact 与本无效闭环运行分开保存。
