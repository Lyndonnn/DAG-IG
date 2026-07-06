# Paper Main v1 Consolidated Results

> Status: superseded old-KL diagnostic. This report consolidates the pre-audit mainline before the KL/checker fixes. It is retained for traceability only. Use `KLFIXED_GRPO_60_REPORT.md`, `PAPER_MAIN_V1_CURRENT_STATUS.md`, and `results/metrics/klfixed_grpo_60_summary.json` for the corrected paper-facing result.

## 1. Scope

This report consolidates the earlier paper-main evidence: Format-SFT baseline, old-KL seed42, old-KL seed43, and fixed-corpus control. It does not reflect the corrected k3 KL reruns or checker v4 headline.

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

These old-KL numbers are diagnostic only. The corrected stable claim is stated in `KLFIXED_GRPO_60_REPORT.md`: KL-fixed two-seed mean strict success is `45.9%` dev and `39.1%` test, versus Format-SFT v4 `40.8%` dev and `34.4%` test.

## 5. Bottleneck

The old-KL bottleneck pattern was retrieval misses plus retrieval-hit-answer-wrong cases. For corrected failure counts, use `KLFIXED_GRPO_60_REPORT.md` and `klfixed_grpo_60_summary.json`.

## 6. Decision

Do not keep the old-KL seed42 checkpoint as the current main checkpoint. The corrected paper-facing result is the KL-fixed seed42/seed43 mean with checker v4 and fixed-reader control.
