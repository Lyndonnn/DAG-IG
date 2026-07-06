# Reproducibility Appendix

## Environment And Commit

- repository: `Lyndonnn/DAG-IG`
- corrected release commits:
  - `30076be` fixes KL/checker/dev-selection reporting;
  - `65356c5` adds KL-fixed fixed-reader control.
- base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- optional local model override: set `DAGIG_LOCAL_3B_MODEL=/path/to/Qwen2.5-VL-3B-Instruct`
- initializer adapter: `outputs/dagig_grpo_main/checkpoints/format_sft`

Do not rely on a hard-coded `/root` HuggingFace snapshot path. The default resolver now uses the public model id unless `DAGIG_LOCAL_3B_MODEL` is explicitly set.

## Data And Corpus

Authoritative manifest:

```text
outputs/dagig_grpo_main/derived_assets/derived_manifest.json
```

Derived split sizes:

| file | rows |
|---|---:|
| `outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl` | 458 |
| `outputs/dagig_grpo_main/derived_assets/grpo_dev.jsonl` | 98 |
| `outputs/dagig_grpo_main/derived_assets/grpo_test.jsonl` | 64 |

Corpus sizes and nature:

| corpus | docs | expected samples | gold-doc coverage | median tokens | use |
|---|---:|---:|---:|---:|---|
| `bm25_train_corpus.jsonl` | 610 | 458 | 91.0% | 6.0 | old-KL diagnostic training |
| `bm25_train_corpus_goldfixed.jsonl` | 610 | 458 | 100.0% | 6.0 | train-only goldfixed control |
| `bm25_eval_corpus.jsonl` | 201 | 162 | 92.6% | 6.0 | dev/test evaluation |

Important boundary: these are frozen Pix2Fact evidence-note corpora with URLs/domains, not live web pages and not live web search. Gold support notes often contain the answer string directly; the dev/test gold-doc answer-string embedded rate is 80.7%. See `results/reports/CORPUS_REALITY_AUDIT.md`.

## Main Training Recipe

The corrected paper-main recipe is:

- reward variant: `paper_main_v1`
- model: Qwen2.5-VL-3B-Instruct
- initializer: Format-SFT adapter
- rollout: two-stage
- loss scope: stage1 only
- GRPO generations: 4
- top-k retrieval: 5
- KL coefficient: 0.1
- KL implementation: non-negative k3 estimator
- learning rate: `1e-6`
- max sequence length: 8192
- stage1 max new tokens: 96
- reader max new tokens: 48
- max samples: 320, using the first 320 train rows
- max optimizer steps: 60
- bf16 and gradient checkpointing enabled

Training summaries:

| run | summary | optimizer steps | micro steps | constant groups | max GPU GB |
|---|---|---:|---:|---:|---:|
| KL-fixed seed42 | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed42/grpo_train_summary.json` | 60 | 240 | 3 | 19.826 |
| KL-fixed seed43 | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed43/grpo_train_summary.json` | 60 | 240 | 1 | 19.842 |

## Current Checkpoints

| role | path |
|---|---|
| Format-SFT baseline | `outputs/dagig_grpo_main/checkpoints/format_sft` |
| KL-fixed seed42 | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed42/checkpoint-60` |
| KL-fixed seed43 | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed43/checkpoint-60` |

## Evaluation

Evaluation script:

```text
scripts/dagig_grpo/05_eval_two_stage.py
```

Common evaluation settings:

- eval corpus: `outputs/dagig_grpo_main/derived_assets/bm25_eval_corpus.jsonl`
- dev file: `outputs/dagig_grpo_main/derived_assets/grpo_dev.jsonl`
- test file: `outputs/dagig_grpo_main/derived_assets/grpo_test.jsonl`
- top-k: 5
- stage1 prompt: compact JSON with `visual_observation` and `search_query`
- reader prompt: compact JSON with `final_answer`
- reader prompt version: `v1`
- checker: answer checker v4

Authoritative current metrics:

| method | dev metric | test metric |
|---|---|---|
| Format-SFT v4 | `outputs/dagig_paper_main_v1/two_stage_metrics_rescored_v4/format_sft_two_stage_own_full_dev.json` | `outputs/dagig_paper_main_v1/two_stage_metrics_rescored_v4/format_sft_two_stage_own_full_test.json` |
| KL-fixed seed42 | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_dev.json` | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_test.json` |
| KL-fixed seed43 | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_dev.json` | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_test.json` |
| KL-fixed seed42 + fixed Format reader | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_formatreader__reader_format_sft_dev.json` | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_formatreader__reader_format_sft_test.json` |
| KL-fixed seed43 + fixed Format reader | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_formatreader__reader_format_sft_dev.json` | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_formatreader__reader_format_sft_test.json` |

Main corrected summary:

```text
results/reports/KLFIXED_GRPO_60_REPORT.md
results/metrics/klfixed_grpo_60_summary.json
```

## Verification

Run:

```bash
python scripts/verify_paper_main_results.py
```

This verifies the corrected result table, KL-fixed summary, fixed-reader control, k3 KL validation, checker-v4 validation, no top-level 7B imports, no hard-coded local model path, and seed42/seed43 training-health records. It does not rerun model inference.
