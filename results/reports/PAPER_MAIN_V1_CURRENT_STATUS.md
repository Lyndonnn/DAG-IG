# Paper Main v1 Current Status

## Current Main Candidate

The current paper-facing candidate is the KL-fixed two-stage GRPO rerun, not the earlier old-KL seed42 checkpoint.

- initializer: `Format-SFT`
- method: two-stage rollout
- stage-1 output: `visual_observation` + `search_query`
- retrieval: frozen BM25 evidence corpus, top-5
- reader: same checkpoint as reader in the main own-reader setting
- policy loss: stage-1 only
- reward: `paper_main_v1`
- KL coefficient: `0.1`
- KL implementation: non-negative k3 estimator
- checker: answer checker v4
- seeds:
  - `paper_main_v1_klfixed_scale60_s320_seed42/checkpoint-60`
  - `paper_main_v1_klfixed_scale60_s320_seed43/checkpoint-60`

## Corrected Main Metrics

| Method | Dev R@5 | Dev strict | Test R@5 | Test strict |
|---|---:|---:|---:|---:|
| Format-SFT v4 | 52.0% | 40.8% | 46.9% | 34.4% |
| KL-fixed GRPO seed42 | 56.1% | 45.9% | 51.6% | 40.6% |
| KL-fixed GRPO seed43 | 56.1% | 45.9% | 48.4% | 37.5% |
| KL-fixed GRPO two-seed mean | 56.1% | 45.9% | 50.0% | 39.1% |

The corrected two-seed mean improves strict success over Format-SFT by `+5.1` dev points and `+4.7` test points.

## Training Health

| Run | Optimizer steps | Micro steps | Constant reward groups | Constant rate |
|---|---:|---:|---:|---:|
| KL-fixed smoke v2 | 1 | 4 | 0 | 0.0% |
| KL-fixed seed42 | 60 | 240 | 3 | 1.2% |
| KL-fixed seed43 | 60 | 240 | 1 | 0.4% |

The earlier `constant_reward_groups ~= 78.8%` concern does not apply to the KL-fixed main reruns.

## Audit Corrections

The external audit identified several issues in the previous result package. Current status:

- KL bug fixed: the training loss now uses k3 KL instead of the old signed log-ratio penalty.
- Checker v4 fixed: AM/PM bare fallback, substring boundary, and numeric-range false positives are blocked.
- Test-selection contamination fixed: the current report uses seed42/seed43 mean and does not select the checkpoint by test performance.
- Old-KL results are diagnostic only.
- The corpus should be described as a frozen Pix2Fact evidence-note corpus, not live web search.
- The method claim should be phrased as node-level DAG-IG reward/credit over a two-stage search agent. Avoid overclaiming true causal counterfactual intervention unless that experiment is explicitly added.

## Evidence Files

- Core fix validation: `results/reports/CORE_FIX_VALIDATION.md`
- Checker-v4 rescore: `results/reports/CHECKER_V4_RESCORING_REPORT.md`
- Dev-selection correction: `results/reports/PAPER_MAIN_V1_CORRECTED_V4_DEV_SELECTION.md`
- Corrected KL-fixed result: `results/reports/KLFIXED_GRPO_60_REPORT.md`
- Machine-readable summary: `results/metrics/klfixed_grpo_60_summary.json`

## Decision

The project has a corrected, cleaner 3B main-result candidate, but it is still a small-sample result with modest paired significance. The next paper-facing experiment should keep the same fixed protocol and add only one stronger variable, such as more GRPO steps or a larger clean training pool, selected by dev protocol.

Do not revive DAG-SFT as the main method. Do not report the old-KL seed42 test result as the final headline.
