# Paper Main v1 Consolidated Results

## 1. Scope

This report consolidates the current paper-main evidence: Format-SFT baseline, seed42 main checkpoint, seed43 confirmation, and fixed-corpus control. It does not introduce new training, new data, or new reward variants.

## 2. Main Table

| Method | Split | R@1 | R@3 | R@5 | Answer correct | Strict success | Format | Retrieval miss | Hit-answer-wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Format-SFT | dev | 35.7% | 49.0% | 52.0% | 45.9% | 42.9% | 100.0% | 47 | 9 |
| Format-SFT | test | 31.2% | 43.8% | 46.9% | 34.4% | 34.4% | 98.4% | 34 | 8 |
| DAG-IG seed42 main | dev | 38.8% | 51.0% | 57.1% | 51.0% | 49.0% | 99.0% | 42 | 8 |
| DAG-IG seed42 main | test | 39.1% | 46.9% | 51.6% | 40.6% | 40.6% | 96.9% | 31 | 7 |
| DAG-IG seed43 confirm | dev | 40.8% | 53.1% | 58.2% | 51.0% | 49.0% | 99.0% | 41 | 9 |
| DAG-IG seed43 confirm | test | 39.1% | 46.9% | 50.0% | 39.1% | 39.1% | 98.4% | 32 | 7 |
| Goldfixed control | dev | 38.8% | 51.0% | 57.1% | 52.0% | 50.0% | 100.0% | 42 | 7 |
| Goldfixed control | test | 35.9% | 45.3% | 50.0% | 39.1% | 39.1% | 96.9% | 32 | 7 |

## 3. Strict Gain/Loss Counts

| Comparison | Common samples | Method-only strict | Baseline-only strict | Both strict | Both fail | Retrieval gain | Retrieval loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed42_vs_format_dev | 98 | 8 | 2 | 40 | 48 | 8 | 3 |
| seed42_vs_format_test | 64 | 5 | 1 | 21 | 37 | 5 | 2 |
| seed43_vs_format_dev | 98 | 7 | 1 | 41 | 49 | 9 | 3 |
| seed43_vs_format_test | 64 | 5 | 2 | 20 | 37 | 5 | 3 |
| goldfixed_vs_format_dev | 98 | 8 | 1 | 41 | 48 | 8 | 3 |
| goldfixed_vs_format_test | 64 | 3 | 0 | 22 | 39 | 3 | 1 |
| goldfixed_vs_seed42_dev | 98 | 2 | 1 | 47 | 48 | 1 | 1 |
| goldfixed_vs_seed42_test | 64 | 1 | 2 | 24 | 37 | 1 | 2 |

## 4. Main Claim Status

The stable claim is that the DAG-IG GRPO recipe improves over Format-SFT in a two-stage offline retrieval setting. Seed42 improves strict success from `42.9%` to `49.0%` on dev and from `34.4%` to `40.6%` on test. Seed43 confirms the recipe with `49.0%` dev and `39.1%` test strict. The fixed-corpus control is train-healthy and reaches `50.0%` dev strict, but does not replace seed42 because test strict is `39.1%`.

## 5. Bottleneck

The remaining bottleneck is not format or answer leakage: format is near-perfect and answer-in-query is zero in the main runs. The dominant errors remain retrieval misses and retrieval-hit-answer-wrong cases. For seed42, dev/test retrieval misses are `42 / 31`, and hit-answer-wrong cases are `8 / 7`. For the fixed-corpus control, dev/test retrieval misses are `42 / 32`, and hit-answer-wrong cases are `7 / 7`.

## 6. Decision

Keep `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60` as the current main checkpoint. Use seed43 as confirmation and the fixed-corpus run as a robustness/control ablation. The next paper-facing step is not another same-recipe GRPO run; it is writing the method/result narrative around node-level DAG-IG credit, then deciding whether a targeted retrieval or reader mechanism is necessary for a stronger final result.
