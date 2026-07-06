# Query-Warm GRPO 30-Step Report

## Purpose

This run tests whether train-only query-node warmup can improve the paper-main v1 two-stage GRPO recipe. It is a controlled mainline experiment, not a new reward variant or reader repair.

## Training Health

| status | optimizer steps | micro steps | constant reward groups | constant rate | max GPU GB | elapsed sec |
|---|---:|---:|---:|---:|---:|---:|
| success | 30 | 120 | 3 | 2.5% | 19.63 | 2516.8 |

The run is healthy: constant-reward groups are low, so this is not the old reward-collapse failure mode.

## Evaluation

| Method | n | R@1 | R@3 | R@5 | answer | strict | format | retrieval miss | hit-answer-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Format-SFT full dev | 98 | 35.7% | 49.0% | 52.0% | 45.9% | 42.9% | 100.0% | 47 | 9 |
| Query-node SFT smoke + fixed reader | 98 | 37.8% | 50.0% | 55.1% | 48.0% | 44.9% | 100.0% | 44 | 10 |
| DAG-IG GRPO seed42 ckpt60 dev | 98 | 38.8% | 51.0% | 57.1% | 51.0% | 49.0% | 99.0% | 42 | 8 |
| DAG-IG GRPO seed43 ckpt60 dev | 98 | 40.8% | 53.1% | 58.2% | 51.0% | 49.0% | 99.0% | 41 | 9 |
| Query-warm GRPO ckpt10 own reader | 98 | 39.8% | 51.0% | 56.1% | 45.9% | 42.9% | 99.0% | 43 | 13 |
| Query-warm GRPO ckpt20 own reader | 98 | 39.8% | 52.0% | 58.2% | 46.9% | 43.9% | 99.0% | 41 | 14 |
| Query-warm GRPO ckpt30 own reader | 98 | 40.8% | 55.1% | 59.2% | 48.0% | 44.9% | 99.0% | 40 | 14 |
| Query-warm GRPO ckpt30 + fixed Format reader | 98 | 40.8% | 55.1% | 59.2% | 51.0% | 48.0% | 100.0% | 40 | 11 |
| DAG-IG GRPO seed42 ckpt60 test | 64 | 39.1% | 46.9% | 51.6% | 40.6% | 40.6% | 96.9% | 31 | 7 |
| DAG-IG GRPO seed43 ckpt60 test | 64 | 39.1% | 46.9% | 50.0% | 39.1% | 39.1% | 98.4% | 32 | 7 |

## Interpretation

- Query-warm ckpt30 improves dev retrieval to `59.2%` own-reader and `59.2%` with fixed Format-SFT reader, the highest dev R@5 among the current dev runs.
- It does not improve final strict success: own-reader strict is `44.9%` and fixed-reader strict is `48.0%`, below the current seed42/seed43 dev strict of `49.0%`.
- Fixed-reader evaluation reduces the answer penalty but still leaves `11` retrieval-hit-answer-wrong cases, so better retrieval alone is not sufficient for promotion.

## Decision

Do not promote query-warm GRPO as the current main checkpoint and do not run test for it. Keep seed42 scale60_s320 checkpoint-60 as the paper-main v1 checkpoint, with seed43 as seed confirmation.

The useful result is diagnostic: query-node supervision can raise retrieval, but the answer node still blocks strict success. The next efficient mainline step is to improve retrieval data for the 124 train samples where existing rollouts contain no hit, and to keep answer-reader changes separate from the current main result.

## Artifacts

- training summary: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_querywarm_stage1loss_kl01_30_s320/grpo_train_summary.json`
- machine summary: `outputs/dagig_paper_main_v1/reports/querywarm_grpo_30_summary.json`
