# DAG-IG Paper Main Evidence Brief

## Paper Position

The main method should be positioned as node-level DAG-IG credit for a two-stage multimodal search agent, optimized with grouped GRPO over the stage-1 policy that emits `visual_observation` and `search_query`. The main evaluation uses each checkpoint as its own reader, and a fixed Format-SFT reader control is reported to isolate query-stage quality.

DAG-SFT is not the main claim. Preference-style planner tuning and query reranking were useful diagnostics, but the paper-facing result is the DAG-IG GRPO agent.

## Main Claim

In the frozen Pix2Fact evidence-note BM25 setting, KL-fixed DAG-IG node-level GRPO improves the Format-SFT two-stage agent on both dev and test as a two-seed mean. The corrected headline does not select the best test seed.

| method | dev R@5 | dev strict | test R@5 | test strict |
|---|---:|---:|---:|---:|
| Format-SFT v4 | 52.0% | 40.8% | 46.9% | 34.4% |
| KL-fixed GRPO seed42 | 56.1% | 45.9% | 51.6% | 40.6% |
| KL-fixed GRPO seed43 | 56.1% | 45.9% | 48.4% | 37.5% |
| KL-fixed GRPO two-seed mean | 56.1% | 45.9% | 50.0% | 39.1% |

## Effect Size

The KL-fixed two-seed mean improves strict success by `+5.1` dev points and `+4.7` test points. Paired comparisons for individual seeds are directionally positive but not conventionally significant, so this should be stated as a modest small-sample result.

## Why The Reward Is Trustworthy

| run | optimizer steps | micro steps | constant groups | constant rate |
|---|---:|---:|---:|---:|
| KL-fixed smoke v2 | 1 | 4 | 0 | 0.0% |
| KL-fixed seed42 | 60 | 240 | 3 | 1.2% |
| KL-fixed seed43 | 60 | 240 | 1 | 0.4% |

The KL-fixed training runs are not collapsed. Earlier reward-component AUC tables remain useful diagnostics, but they should not be described as causal counterfactual intervention evidence.

## Reproducibility Anchors

- KL-fixed seed42: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed42/checkpoint-60`
- KL-fixed seed43: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed43/checkpoint-60`
- corrected result report: `results/reports/KLFIXED_GRPO_60_REPORT.md`
- core fix validation: `results/reports/CORE_FIX_VALIDATION.md`
- corpus boundary: `results/reports/CORPUS_REALITY_AUDIT.md`
- machine-readable summary: `results/metrics/klfixed_grpo_60_summary.json`

## Claim Boundaries

- Do not claim DAG-SFT is the main method; it is a diagnostic/pretraining baseline.
- Do not report old-KL seed42 as the final headline; it is diagnostic only.
- Do not select a checkpoint by test performance; report the KL-fixed two-seed mean.
- Do not claim real web search generalization; all results use the frozen offline Pix2Fact evidence-note BM25 corpus.
- Do not call the corpus noisy full web documents; the dev/test corpus has 201 short evidence notes with median length 6 tokens.
- Do not claim causal counterfactual intervention unless explicit intervention experiments are added.
- Do not claim answer extraction is solved. The remaining bottlenecks are retrieval misses and retrieval-hit-answer-wrong cases.
- Do not launch more same-recipe GRPO runs without a new mechanism. The current next step is paper writing plus targeted qualitative/error presentation.

## Paper-Completion Next Step

Write the paper around the corrected evidence chain: problem formulation, DAG-IG node credit, two-stage GRPO training with k3 KL, checker v4, KL-fixed two-seed mean, fixed-reader control, corpus boundary, and failure analysis. Additional experiments should be limited to paper-essential presentation gaps, not new method branches.
