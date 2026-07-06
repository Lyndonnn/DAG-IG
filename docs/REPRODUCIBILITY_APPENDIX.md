# Reproducibility Appendix

## Environment And Commit

- project root: `/root/autodl-tmp/search-test-1`
- git commit recorded for the 414/main run family: `f8cf0e78bf9768638105041291645208e8474154`
- base model path used by run configs:
  `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3`
- initializer adapter:
  `outputs/dagig_grpo_main/checkpoints/format_sft`

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

Corpus sizes:

| corpus | docs | use |
|---|---:|---|
| `outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl` | 610 | seed42/seed43 main training |
| `outputs/dagig_grpo_main/derived_assets/bm25_eval_corpus.jsonl` | 201 | dev/test evaluation |
| `outputs/dagig_paper_main_v1/derived_assets/bm25_train_corpus_goldfixed.jsonl` | 610 | train-only goldfixed control |

The original downloaded asset had empty BM25 corpus files and empty warmup SFT file. The derived files were built only from package-local source/grpo/evidence fields and did not mutate the source asset.

## Main Training Recipe

The paper-main recipe is:

- reward variant: `paper_main_v1`
- model: Qwen2.5-VL-3B-Instruct
- initializer: Format-SFT adapter
- rollout: two-stage
- loss scope: stage1 only
- GRPO generations: 4
- top-k retrieval: 5
- KL coefficient: 0.1
- learning rate: `1e-6`
- max sequence length: 8192
- stage1 max new tokens: 96
- reader max new tokens: 48
- max samples: 320
- max optimizer steps: 60
- bf16 and gradient checkpointing enabled

Training summaries:

| run | summary | optimizer steps | micro steps | constant groups | max GPU GB |
|---|---|---:|---:|---:|---:|
| seed42 main | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/grpo_train_summary.json` | 60 | 240 | 2 | 19.828 |
| seed43 confirm | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/grpo_train_summary.json` | 60 | 240 | 2 | 19.825 |
| goldfixed control | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/grpo_train_summary.json` | 60 | 240 | 2 | 19.833 |

## Current Checkpoints

| role | path |
|---|---|
| Format-SFT baseline | `outputs/dagig_grpo_main/checkpoints/format_sft` |
| seed42 main | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60` |
| seed43 confirmation | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/checkpoint-60` |
| goldfixed control | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/checkpoint-60` |

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

Authoritative current metrics:

| method | dev metric | test metric |
|---|---|---|
| Format-SFT | `outputs/dagig_paper_main_v1/two_stage_metrics_rescored_v3/format_sft_two_stage_own_full_dev.json` | `outputs/dagig_paper_main_v1/two_stage_metrics_rescored_v3/format_sft_two_stage_own_full_test.json` |
| seed42 main | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.json` | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.json` |
| seed43 confirm | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.json` | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test.json` |
| goldfixed control | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_goldfixed_scale60_s320_ckpt60_dev.json` | `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_goldfixed_scale60_s320_ckpt60_test.json` |

The main table uses parser/checker v3 rescoring for the Format-SFT baseline. The seed42/seed43/goldfixed metric files above are the current paper-main metrics.

## Reward Audit And Consolidation

Main reward/component reports:

```text
outputs/dagig_paper_main_v1/reports/node_credit_component_analysis/NODE_CREDIT_COMPONENT_ANALYSIS.md
outputs/dagig_paper_main_v1/reports/PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md
outputs/dagig_paper_main_v1/reports/SEED_CONFIRMATION_REPORT.md
outputs/dagig_paper_main_v1/reports/GOLDFIXED_GRPO_60_REPORT.md
```

Regenerate paper-facing result assets:

```bash
python scripts/dagig_paper_main/25_consolidate_main_results.py
python scripts/dagig_paper_main/26_analyze_node_credit_components.py
python scripts/dagig_paper_main/27_build_paper_experiment_package.py
python scripts/dagig_paper_main/28_build_paper_case_studies.py
```

## Command File

Runnable command templates are collected in:

```text
outputs/dagig_paper_main_v1/paper_assets/reproduce_main_commands.sh
```

These commands are intended for controlled reproduction. Running the training commands will overwrite output directories if the same paths are reused, so copy them to a new output root when preserving the current run is important.
