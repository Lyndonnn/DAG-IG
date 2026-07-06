# Gold-Fixed GRPO 60 Report

## 1. Motivation

The train BM25 corpus had missing `is_gold=true` labels for 41 train samples. After a uniform train-only gold-label fix, the reward was re-audited and passed. This run tests the same stable paper-main recipe with the fixed train corpus as the only intended protocol change.

## 2. Reward Health

- reward AUC vs fixed-corpus hit: `0.999`
- reward AUC vs fixed-corpus strict: `0.938`
- audit constant reward groups: `37 / 3664` (`1.0%`)

## 3. Training

- checkpoint root: `/root/autodl-tmp/search-test-1/outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320`
- optimizer steps: `60`
- micro steps: `240`
- training constant reward groups: `2 / 240` (`0.83%`)
- max GPU memory: `19.833` GB

## 4. Dev Sweep

| Method | Dev R@5 | Dev answer | Dev strict | Format | Retrieval miss | Hit-answer-wrong |
|---|---:|---:|---:|---:|---:|---:|
| Format-SFT | 52.0% | 45.9% | 42.9% | 100.0% | 47 | 9 |
| Current seed42 ckpt60 | 57.1% | 51.0% | 49.0% | 99.0% | 42 | 8 |
| Seed43 ckpt60 | 58.2% | 51.0% | 49.0% | 99.0% | 41 | 9 |
| Goldfixed ckpt20 | 52.0% | 45.9% | 43.9% | 100.0% | 47 | 8 |
| Goldfixed ckpt40 | 55.1% | 48.0% | 45.9% | 100.0% | 44 | 9 |
| Goldfixed ckpt60 | 57.1% | 52.0% | 50.0% | 100.0% | 42 | 7 |

## 5. Test Check

| Method | Test R@5 | Test answer | Test strict | Format | Retrieval miss | Hit-answer-wrong |
|---|---:|---:|---:|---:|---:|---:|
| Format-SFT | 46.9% | 34.4% | 34.4% | 98.4% | 34 | 8 |
| Current seed42 ckpt60 | 51.6% | 40.6% | 40.6% | 96.9% | 31 | 7 |
| Seed43 ckpt60 | 50.0% | 39.1% | 39.1% | 98.4% | 32 | 7 |
| Goldfixed ckpt60 | 50.0% | 39.1% | 39.1% | 96.9% | 32 | 7 |

## 6. Decision

NO PROMOTION. The fixed-corpus rerun is train-healthy and improves dev strict to `50.0%`, but test strict is `39.1%`, below the current seed42 main checkpoint's `40.6%`. Test R@5 also drops from `51.6%` to `50.0%`. Keep the existing seed42 scale60_s320 checkpoint as the paper-main checkpoint, and treat the fixed-corpus run as a useful robustness/control run rather than the new main result.

## 7. Next Mainline Action

Do not run another same-recipe GRPO immediately. The useful signal from the fixed corpus is that reward health is solid, but the generalization bottleneck remains retrieval coverage and answer extraction. The next paper-facing step should be a targeted comparison/report section: main seed42 result, seed43 confirmation, fixed-corpus control, and failure categories. Only after that should another method change be attempted.
