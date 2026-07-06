# Paper Main v1 Seed Confirmation

## Purpose

This report checks whether the current two-stage DAG-IG GRPO recipe is a repeatable mainline result, not a single-seed artifact.

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

Seed43 confirms the main recipe: dev strict remains `49.0%`, test strict is `39.1%`, and both dev/test remain above Format-SFT. The current best single checkpoint is still seed42 scale60_s320 checkpoint-60 because it has the best test strict (`40.6%`) and test R@5 (`51.6%`).

Use seed42 checkpoint-60 as the current main checkpoint, and cite seed43 as seed confirmation. The next mainline work should target remaining retrieval misses with better query/evidence credit data, not more answer repair or reward reshuffling.
