# DAG-IG Paper Main v1 Reward Audit

## 1. Protocol

- Frozen rollout schema: `protocol/PAPER_MAIN_V1_SCHEMA.md`
- Derived asset: `outputs/dagig_grpo_main/derived_assets`
- Train corpus: `outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl`
- This audit does not train. It rescored existing sampled train rollouts with dense node-level DAG-IG credits.

## 2. Counts

- rollouts: `14656`
- groups: `3664`
- constant reward groups: `40` = `1.1%`
- reward mean/std/min/max: `0.2855` / `0.2963` / `-0.2400` / `1.1600`

## 3. Component Coverage

- `format_credit` nonzero: `98.2%`
- `visual_credit` nonzero: `89.6%`
- `query_credit` nonzero: `45.7%`
- `evidence_credit` nonzero: `41.8%`
- `answer_credit` nonzero: `2.1%`
- `leak_penalty` nonzero: `0.3%`
- `path_penalty` nonzero: `1.0%`

## 4. Predictiveness

- `query_credit_auc_support_hit`: `1.000`
- `evidence_credit_auc_support_hit`: `1.000`
- `answer_credit_auc_answer_correct`: `1.000`
- `total_reward_auc_strict`: `0.939`

## 5. Gates

- `max_constant_group_rate`: `0.3`
- `constant_group_gate`: `True`
- `min_query_auc`: `0.7`
- `query_auc_gate`: `True`
- `min_total_auc`: `0.7`
- `total_auc_gate`: `True`

## 6. Decision

- status: `go`
- Reward signal is sufficiently non-constant for a small GRPO smoke using this paper-main v1 reward.
