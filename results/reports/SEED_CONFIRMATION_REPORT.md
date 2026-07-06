# Paper Main v1 Seed Confirmation

> Status: superseded old-KL diagnostic. This report records the pre-audit seed confirmation run and is retained for traceability only. It must not be used as the corrected paper headline. Use `KLFIXED_GRPO_60_REPORT.md`, `PAPER_MAIN_V1_CURRENT_STATUS.md`, and `results/metrics/klfixed_grpo_60_summary.json` for the current KL-fixed two-seed result.

## Purpose

This report checks whether the earlier two-stage DAG-IG GRPO recipe was repeatable before the reviewer-audit KL/checker fixes.

## Training Health

| Run | status | optimizer steps | micro steps | constant reward groups | constant rate | max GPU GB | elapsed sec |
|---|---|---:|---:|---:|---:|---:|---:|
| seed42 | success | 60 | 240 | 2 | 0.8% | 19.828 | 5706.6 |
| seed43 | success | 60 | 240 | 2 | 0.8% | 19.825 | 5929.3 |

Both runs avoid the old constant-reward failure mode. The reward signal remains usable under the paper-main v1 two-stage setup.

## Seed43 Dev Checkpoint Sweep

| Checkpoint | Dev R@1 | Dev R@3 | Dev R@5 | Dev answer | Dev strict | Format parse | Retrieval miss | Hit-answer-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed43 ckpt20 | 37.8% | 49.0% | 55.1% | 50.0% | 48.0% | 100.0% | 44 | 7 |
| seed43 ckpt40 | 36.7% | 50.0% | 57.1% | 50.0% | 49.0% | 99.0% | 42 | 8 |
| seed43 ckpt60 | 40.8% | 53.1% | 58.2% | 51.0% | 49.0% | 99.0% | 41 | 9 |

Selection rule: choose by dev strict first, then R@5 as a tie-breaker. ckpt40 and ckpt60 tie on dev strict; ckpt60 has higher R@5, so ckpt60 was evaluated on test.

## Main Comparison

| Method | Dev R@5 | Dev answer | Dev strict | Dev format | Dev hit-answer-wrong | Test R@5 | Test answer | Test strict | Test format | Test hit-answer-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Format-SFT | 52.0% | 45.9% | 42.9% | 100.0% | 9 | 46.9% | 34.4% | 34.4% | 98.4% | 8 |
| DAG-IG GRPO seed42 ckpt60 | 57.1% | 51.0% | 49.0% | 99.0% | 8 | 51.6% | 40.6% | 40.6% | 96.9% | 7 |
| DAG-IG GRPO seed43 ckpt60 | 58.2% | 51.0% | 49.0% | 99.0% | 9 | 50.0% | 39.1% | 39.1% | 98.4% | 7 |

## Gains Over Format-SFT

| Run | Dev R@5 gain | Dev strict gain | Test R@5 gain | Test strict gain |
|---|---:|---:|---:|---:|
| seed42 ckpt60 | 5.1% | 6.1% | 4.7% | 6.2% |
| seed43 ckpt60 | 6.1% | 6.1% | 3.1% | 4.7% |
| two-seed mean | 5.6% | 6.1% | 3.9% | 5.5% |

## Decision

This old-KL diagnostic showed directionally positive seed behavior, but it is no longer the paper-facing selection result. The corrected report uses k3 KL, checker v4, and the seed42/seed43 mean instead of selecting the best test checkpoint.

Use this file only as a historical diagnostic. Do not use seed42 checkpoint-60 from this report as the current main checkpoint.
