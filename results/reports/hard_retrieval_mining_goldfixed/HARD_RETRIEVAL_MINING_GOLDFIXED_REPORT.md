# Gold-Fixed Hard Retrieval Mining Report

## Scope

This rebuilds train query hit-vs-miss mining after fixing train corpus gold labels. It uses the same existing rollout text; only retrieval hit labels are recomputed with the fixed train corpus.

## Train Rollout Coverage

- train rollouts: `14656`
- train samples: `458`
- train query hit-vs-miss pairs: `339`
- `already_hit_no_miss_pair`: `26`
- `candidate_insufficient_no_hit_rollout`: `93`
- `learnable_from_existing_rollouts`: `339`

## Source Distribution

| source | rollouts | hit@5 rollouts | strict rollouts approx |
|---|---:|---:|---:|
| dagig_grpo_full | 3664 | 1697 | 38 |
| dagig_grpo_no_visual | 3664 | 1724 | 29 |
| outcome_grpo | 3664 | 1605 | 31 |
| trajectory_grpo | 3664 | 1668 | 31 |

## Decision

Use these goldfixed pairs for future query-node warmup or candidate analysis. The previous pair file is still useful for reproducing old runs, but future train-side work should use this fixed version.

## Artifacts

- groups: `outputs/dagig_paper_main_v1/reports/hard_retrieval_mining_goldfixed/train_hard_retrieval_groups_goldfixed.jsonl`
- pairs: `outputs/dagig_paper_main_v1/reports/hard_retrieval_mining_goldfixed/train_query_hit_vs_miss_pairs_goldfixed.jsonl`
- summary: `outputs/dagig_paper_main_v1/reports/hard_retrieval_mining_goldfixed/hard_retrieval_mining_goldfixed_summary.json`
