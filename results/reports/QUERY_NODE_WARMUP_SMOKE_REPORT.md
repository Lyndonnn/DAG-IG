# Query Node Warmup Smoke Report

## Scope

This tests whether train-only query hit-vs-miss pairs can supervise stage-1 query generation. It is not a final training run and does not use dev/test labels for training.

## Data

- train query-node rows: `311`
- skipped rows: `0`
- assistant target fields: `visual_observation`, `search_query`
- final answer included in target: `False`
- dev/test labels used for training: `False`

## Training Smoke

- init adapter: `outputs/dagig_grpo_main/checkpoints/format_sft`
- output adapter: `outputs/dagig_paper_main_v1/checkpoints/query_node_sft_format_init_smoke20`
- max steps: `20`
- train log: `outputs/dagig_paper_main_v1/eval_logs/query_node_sft_format_init_smoke20_train.log`

## Evaluation

Evaluation uses the query-node adapter for stage 1 and a fixed Format-SFT reader, so movement is mostly query/retrieval-side rather than reader drift.

| Method | n | R@1 | R@3 | R@5 | answer | strict | format | retrieval miss | hit-answer-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Format-SFT full dev | 98 | 35.7% | 49.0% | 52.0% | 45.9% | 42.9% | 100.0% | 47 | 9 |
| Query-node SFT smoke dev20 + fixed reader | 20 | 15.0% | 45.0% | 60.0% | 40.0% | 35.0% | 100.0% | 8 | 5 |
| Query-node SFT smoke full dev + fixed reader | 98 | 37.8% | 50.0% | 55.1% | 48.0% | 44.9% | 100.0% | 44 | 10 |
| DAG-IG GRPO seed42 full dev | 98 | 38.8% | 51.0% | 57.1% | 51.0% | 49.0% | 99.0% | 42 | 8 |
| DAG-IG GRPO seed43 full dev | 98 | 40.8% | 53.1% | 58.2% | 51.0% | 49.0% | 99.0% | 41 | 9 |

## Decision

The query-node SFT smoke passes infrastructure and improves over Format-SFT on full-dev R@5 (`55.1%` vs `52.0%`) and strict (`44.9%` vs `42.9%`). It does not beat the current GRPO checkpoints (`49.0%` dev strict), so it should not be promoted as a standalone method.

Use it only as a candidate initialization or auxiliary warmup for the next GRPO iteration. The next mainline experiment should test whether GRPO initialized from this query-node warmup reduces retrieval misses beyond the current seed42/seed43 recipe.
