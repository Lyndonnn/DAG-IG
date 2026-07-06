# DAG-IG Pix2Fact Core Repository

This repository is the cleaned paper-core package for:

**DAG-IG: Counterfactual Node-Level Credit Assignment for Long-Horizon Multimodal Search Agents**

The main experiment studies a two-stage Pix2Fact search agent:

```text
image + question
-> visual_observation
-> search_query
-> retrieve top-k evidence
-> final_answer
```

DAG-IG assigns counterfactual credit to the visual, query, evidence, and answer nodes, then uses the node-level reward for GRPO training.

## Current Main Result

The paper-main 3B result is:

- initializer: `Format-SFT`
- method: two-stage rollout
- policy loss: stage-1 only
- reward: `paper_main_v1`
- KL: `0.1`
- main checkpoint: `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60`

Parser/checker v3 metrics:

| Method | Dev R@5 | Dev strict | Test R@5 | Test strict |
|---|---:|---:|---:|---:|
| Format-SFT | 52.0% | 42.9% | 46.9% | 34.4% |
| DAG-IG medium30 ckpt30 | 57.1% | 48.0% | 50.0% | 39.1% |
| DAG-IG scale60_s320 seed42 ckpt60 | 57.1% | 49.0% | 51.6% | 40.6% |
| DAG-IG scale60_s320 seed43 ckpt60 | 58.2% | 49.0% | 50.0% | 39.1% |

The main result improves strict success over Format-SFT by +6.1 dev points and +6.2 test points. Seed43 confirms the recipe is not a single-seed artifact.

## Repository Layout

```text
scripts/
  dagig_grpo/          Core GRPO training and two-stage evaluation code
  dagig_paper_main/    Paper-main result consolidation and audits
  reproduce_main_commands.sh

results/
  reports/             Main reports and diagnostic summaries
  metrics/             Consolidated JSON metric summaries
  tables/              Main paper tables as CSV and TeX
  case_studies/        Case-study summary and category counts

paper/
  latex/               LaTeX source scaffold
  figures/             Method/reward TeX figure snippets
  tables/              Paper table TeX files
  venue_template_parts/

docs/
  PAPER_MAIN_V1_SCHEMA.md
  REPRODUCIBILITY_APPENDIX.md
  PAPER_DRAFT_V0.md
  reviewer/readiness docs
```

## What Is Not Included

This GitHub-core package intentionally excludes:

- model checkpoints and LoRA adapter weights;
- Pix2Fact images and raw datasets;
- generated prediction JSONL dumps;
- intermediate failed/debug branches;
- local caches, logs, and compiled LaTeX artifacts.

The main LoRA adapter file is larger than GitHub's normal 100MB file limit. Release it separately via Git LFS, Hugging Face, or cloud storage if public checkpoint release is needed.

## Key Files

- Main status: `results/reports/PAPER_MAIN_V1_CURRENT_STATUS.md`
- Consolidated results: `results/reports/PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`
- Reward audit: `results/reports/REWARD_AUDIT_REPORT.md`
- Seed confirmation: `results/reports/SEED_CONFIRMATION_REPORT.md`
- Goldfixed control: `results/reports/GOLDFIXED_GRPO_60_REPORT.md`
- Evidence chain: `results/reports/MAINLINE_EVIDENCE_CHAIN.md`
- Reproduction appendix: `docs/REPRODUCIBILITY_APPENDIX.md`
- Commands: `scripts/reproduce_main_commands.sh`

## Reproduction Notes

The command template in `scripts/reproduce_main_commands.sh` assumes the original local asset layout from the experiment machine. For a fresh machine, update:

- base model path;
- Format-SFT adapter path;
- train/dev/test JSONL paths;
- BM25 train/eval corpus paths;
- output root.

The core training/evaluation scripts are included, but this repo does not bundle the full dataset or checkpoints.

## Main Claim Boundary

The main method is **DAG-IG GRPO over a two-stage multimodal search agent**.

Do not present DAG-SFT as the main method. Do not present `reward_v3` verifier shaping as the 7B main method. The 7B extension uses `paper_main_v1` as the main reward for same-backbone comparison; verifier-shaped reward variants are optional ablations only.
