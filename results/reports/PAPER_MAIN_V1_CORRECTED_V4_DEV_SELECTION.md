# Paper Main v1 Corrected v4 Dev-Selection Report

## Scope

This report corrects existing paper-main predictions for answer checker v4 and removes test-set checkpoint selection. It does not retrain models and it does not repair the old training-time KL bug. Therefore these old-KL checkpoints are diagnostic only until KL-fixed GRPO is rerun.

## v4 Main Table

| Method | Split | n | R@5 | Answer correct | Strict success | Strict gain vs Format | Retrieval miss | Hit-answer-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Format-SFT baseline | dev | 98 | 52.0% | 42.9% | 40.8% | - | 47 | 11 |
| Format-SFT baseline | test | 64 | 46.9% | 34.4% | 34.4% | - | 34 | 8 |
| DAG-IG old-KL seed42 | dev | 98 | 57.1% | 49.0% | 46.9% | +6.1 | 42 | 10 |
| DAG-IG old-KL seed42 | test | 64 | 51.6% | 40.6% | 40.6% | +6.2 | 31 | 7 |
| DAG-IG old-KL seed43 | dev | 98 | 58.2% | 49.0% | 46.9% | +6.1 | 41 | 11 |
| DAG-IG old-KL seed43 | test | 64 | 50.0% | 39.1% | 39.1% | +4.7 | 32 | 7 |
| Goldfixed train-corpus control | dev | 98 | 57.1% | 50.0% | 48.0% | +7.1 | 42 | 9 |
| Goldfixed train-corpus control | test | 64 | 50.0% | 39.1% | 39.1% | +4.7 | 32 | 7 |
| DAG-IG seed42 with fixed Format-SFT reader | dev | 98 | 57.1% | 48.0% | 45.9% | +5.1 | 42 | 11 |
| DAG-IG seed42 with fixed Format-SFT reader | test | 64 | 51.6% | 40.6% | 40.6% | +6.2 | 31 | 7 |

## Dev-Only Selection

- Candidate runs considered for single-checkpoint selection: `seed42`, `seed43`, `goldfixed`.
- Selection rule: highest dev strict, then dev R@5. Test metrics are not used.
- Selected by this rule: `goldfixed` (Goldfixed train-corpus control).
- Selected dev/test strict: `48.0%` / `39.1%`.
- The old headline that chose seed42 because of higher test strict is not protocol-clean and should not be used.

## Two-Seed Mean

- seed42/seed43 mean dev strict: `46.9%`; mean dev R@5: `57.7%`.
- seed42/seed43 mean test strict: `39.8%`; mean test R@5: `50.8%`.
- Mean strict gain vs Format-SFT: dev `+6.1` points, test `+5.5` points.

## Paired Strict Comparisons vs Format-SFT

| Comparison | Split | common | method-only strict | format-only strict | McNemar exact p | R@5 gain | R@5 loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| DAG-IG old-KL seed42 | dev | 98 | 8 | 2 | 0.109 | 8 | 3 |
| DAG-IG old-KL seed42 | test | 64 | 5 | 1 | 0.219 | 5 | 2 |
| DAG-IG old-KL seed43 | dev | 98 | 7 | 1 | 0.070 | 9 | 3 |
| DAG-IG old-KL seed43 | test | 64 | 5 | 2 | 0.453 | 5 | 3 |
| Goldfixed train-corpus control | dev | 98 | 8 | 1 | 0.039 | 8 | 3 |
| Goldfixed train-corpus control | test | 64 | 3 | 0 | 0.250 | 3 | 1 |
| DAG-IG seed42 with fixed Format-SFT reader | dev | 98 | 6 | 1 | 0.125 | 8 | 3 |
| DAG-IG seed42 with fixed Format-SFT reader | test | 64 | 5 | 1 | 0.219 | 5 | 2 |

## Decision

Use this report as the corrected status for old predictions only. The answer-checker false positives are fixed, and the test-set selection issue is removed. However, because the old GRPO checkpoints were trained with the incorrect signed KL penalty, they are not final paper-main results. The next required experiment is a KL-fixed rerun of the same recipe, reported with checker v4 and dev-only selection/two-seed mean.
