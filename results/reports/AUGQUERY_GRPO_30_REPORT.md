# Augmented-Query GRPO 30-Step Report

## Scope

This is the dev-gated GRPO run initialized from augmented query-node SFT. It uses the same paper-main v1 two-stage stage1-loss recipe as the current mainline. It is evaluated on dev only unless it beats the current dev gate.

## Training Health

| status | optimizer steps | micro steps | constant reward groups | constant rate | max GPU GB | elapsed sec |
|---|---:|---:|---:|---:|---:|---:|
| success | 30 | 120 | 0 | 0.0% | 19.81 | 2461.9 |

The run is healthy: `constant_reward_groups=0`, so this is not reward collapse.

## Dev Evaluation

| Method | n | R@1 | R@3 | R@5 | answer | strict | format | retrieval miss | hit-answer-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Format-SFT full dev | 98 | 35.7% | 49.0% | 52.0% | 45.9% | 42.9% | 100.0% | 47 | 9 |
| Augmented query-node SFT smoke + fixed reader | 98 | 43.9% | 53.1% | 56.1% | 49.0% | 48.0% | 100.0% | 43 | 8 |
| DAG-IG GRPO seed42 ckpt60 dev | 98 | 38.8% | 51.0% | 57.1% | 51.0% | 49.0% | 99.0% | 42 | 8 |
| DAG-IG GRPO seed43 ckpt60 dev | 98 | 40.8% | 53.1% | 58.2% | 51.0% | 49.0% | 99.0% | 41 | 9 |
| Query-warm GRPO ckpt30 + fixed Format reader | 98 | 40.8% | 55.1% | 59.2% | 51.0% | 48.0% | 100.0% | 40 | 11 |
| Aug-query GRPO ckpt10 dev | 98 | 42.9% | 53.1% | 55.1% | 48.0% | 45.9% | 99.0% | 44 | 9 |
| Aug-query GRPO ckpt20 dev | 98 | 42.9% | 54.1% | 56.1% | 48.0% | 45.9% | 99.0% | 43 | 10 |
| Aug-query GRPO ckpt30 dev | 98 | 42.9% | 54.1% | 56.1% | 48.0% | 45.9% | 99.0% | 43 | 10 |

## Interpretation

- Best augmented-init GRPO checkpoint is `Aug-query GRPO ckpt20 dev` with dev strict `45.9%` and R@5 `56.1%`.
- Current main seed42 ckpt60 remains better: dev strict `49.0%`, R@5 `57.1%`.
- Augmented SFT data was useful as a warmup, but GRPO from that warmup degraded final dev strict rather than improving it.
- Because the dev gate failed, do not run test for this checkpoint and do not promote it to the paper-main result.

## Decision

No-go for augmented-init GRPO. Keep `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60` as the current main checkpoint, with seed43 as confirmation. The next mainline action should target stronger non-oracle query candidate generation for the remaining no-hit train samples, not another GRPO run from the same augmented warmup.

## Artifacts

- training summary: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_augquery_stage1loss_kl01_30_s320/grpo_train_summary.json`
- dev metrics: `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_augquery_stage1loss_kl01_30_s320_ckpt{10,20,30}_dev.json`
- machine summary: `outputs/dagig_paper_main_v1/reports/augquery_grpo_30_summary.json`
