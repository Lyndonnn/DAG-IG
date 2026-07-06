# KL-Fixed GRPO 60-Step Audit Report

## 1. Scope

This report fixes the reviewer audit issues that affect the main GRPO result: the KL penalty is now the non-negative k3 estimator, answer matching is rescored with checker v4, and model selection is reported without choosing by test performance.

Old-KL GRPO numbers are kept only as diagnostics. The paper-facing candidate is the KL-fixed rerun.

## 2. Core Fix Validation

- validation passed: `True`
- k3 KL same-policy value: `0.0`
- k3 KL positive case: `0.3678794503211975`
- k3 gradient nonzero check: `0.031606025993824005`
- bf16 near-zero KL after clamp: `0.0`
- checker v4 blocks the audited false positives: bare AM/PM fallback, substring boundary errors, and numeric-range-as-single-answer errors.

## 3. Training Stability

| run | optimizer steps | micro steps | constant groups | constant rate | max GPU GB |
|---|---:|---:|---:|---:|---:|
| klfixed_smoke_v2 | 1 | 4 | 0 | 0.0% | 18.362 |
| klfixed_seed42 | 60 | 240 | 3 | 1.2% | 19.826 |
| klfixed_seed43 | 60 | 240 | 1 | 0.4% | 19.842 |

The earlier constant-reward concern does not hold for the KL-fixed main reruns: seed42 has 3/240 constant groups and seed43 has 1/240.

## 4. Main Metrics

| Method | Dev R@5 | Dev strict | Test R@5 | Test strict |
|---|---:|---:|---:|---:|
| Format-SFT baseline | 52.0% | 40/98 = 40.8% | 46.9% | 22/64 = 34.4% |
| old-KL GRPO seed42 diagnostic | 57.1% | 46/98 = 46.9% | 51.6% | 26/64 = 40.6% |
| old-KL GRPO seed43 diagnostic | 58.2% | 46/98 = 46.9% | 50.0% | 25/64 = 39.1% |
| KL-fixed GRPO seed42 | 56.1% | 45/98 = 45.9% | 51.6% | 26/64 = 40.6% |
| KL-fixed GRPO seed43 | 56.1% | 45/98 = 45.9% | 48.4% | 24/64 = 37.5% |

## 5. Two-Seed Mean

| Method | Dev R@5 | Dev strict | Test R@5 | Test strict | Dev strict gain vs Format | Test strict gain vs Format |
|---|---:|---:|---:|---:|---:|---:|
| old-KL two-seed mean diagnostic | 57.7% | 46.9% | 50.8% | 39.8% | +6.1 | +5.5 |
| KL-fixed two-seed mean | 56.1% | 45.9% | 50.0% | 39.1% | +5.1 | +4.7 |

KL-fixed mean strict success is dev `45.9%` and test `39.1%`, versus Format-SFT dev `40.8%` and test `34.4%`. The corrected gain is therefore +5.1 dev and +4.7 test points.

## 6. Paired Significance

| Method | Split | method-only strict | baseline-only strict | McNemar exact p | R@5 gains/losses |
|---|---|---:|---:|---:|---|
| old-KL GRPO seed42 diagnostic | dev | 8 | 2 | 0.1094 | +8 / -3 |
| old-KL GRPO seed42 diagnostic | test | 5 | 1 | 0.2188 | +5 / -2 |
| old-KL GRPO seed43 diagnostic | dev | 7 | 1 | 0.0703 | +9 / -3 |
| old-KL GRPO seed43 diagnostic | test | 5 | 2 | 0.4531 | +5 / -3 |
| KL-fixed GRPO seed42 | dev | 7 | 2 | 0.1797 | +8 / -4 |
| KL-fixed GRPO seed42 | test | 5 | 1 | 0.2188 | +5 / -2 |
| KL-fixed GRPO seed43 | dev | 6 | 1 | 0.1250 | +7 / -3 |
| KL-fixed GRPO seed43 | test | 2 | 0 | 0.5000 | +2 / -1 |

The paired tests are directionally positive but not conventionally significant for KL-fixed seed42/seed43. This should be described as a small-sample main candidate, not a settled large-scale result.

## 7. Corrected Interpretation

- The old KL penalty was invalid for the paper claim; old-KL results should be marked diagnostic only.
- The corrected KL-fixed rerun keeps the same direction of improvement over Format-SFT under checker v4.
- Seed42 alone matches the old test headline, but seed43 is lower; the clean headline is the two-seed mean, not best test seed.
- The result is useful enough to continue the DAG-IG main line, but the paper should avoid overstating statistical certainty until more seeds or larger data are run.

## 8. Next Step

Do not start unrelated DPO/RL variants before closing this main path. The next efficient step is to run the same KL-fixed recipe with one stronger setting only: either more GRPO steps or a larger clean training pool, selected by dev protocol, while keeping checker v4 and k3 KL fixed.
