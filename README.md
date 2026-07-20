# DAG-IG Pix2Fact Core Repository

> **Current status (2026-07-20):** the repository now includes the v6 complete
> counterfactual-DAG development snapshot. The 3B KL-fixed GRPO result below is
> a historical baseline, not the final DAG-IG proof. A support-label audit
> invalidated the legacy support/strict contract, so the corrected full
> selector-only experiment is still in progress. Read
> [`docs/CURRENT_RESEARCH_STATUS_2026-07-20.md`](docs/CURRENT_RESEARCH_STATUS_2026-07-20.md)
> before using any result as a paper claim.

This repository is the cleaned paper-core package for:

**DAG-IG: Node-Level Credit Assignment for Long-Horizon Multimodal Search Agents**

The main experiment studies a two-stage Pix2Fact search agent:

```text
image + question
-> visual_observation
-> search_query
-> retrieve top-k evidence
-> final_answer
```

DAG-IG assigns node-level credit/reward to the visual, query, evidence, and answer nodes, then uses that signal for GRPO training. The released 3B result should be described as node-level credit optimization; stronger counterfactual-causal claims require additional intervention experiments.

## Historical 3B Result Status

Important: this repository has been updated after an external audit. The old
parser/checker-v3 and old-KL headline is retained only as diagnostic history.
The paper-facing result now uses:

- k3 KL penalty, not the old signed log-ratio penalty;
- answer checker v4;
- dev-only / two-seed reporting, with no test-set checkpoint selection;
- a two-stage rollout evaluation: `visual_observation + search_query`, BM25 retrieval, then final answer reading.

The historical corrected 3B candidate is:

- initializer: `Format-SFT`
- method: two-stage rollout
- policy loss: stage-1 only
- reward: `paper_main_v1`
- KL coefficient: `0.1`, implemented with non-negative k3 KL
- checkpoints:
  - `paper_main_v1_klfixed_scale60_s320_seed42/checkpoint-60`
  - `paper_main_v1_klfixed_scale60_s320_seed43/checkpoint-60`

Checker-v4 / KL-fixed metrics:

| Method | Dev R@5 | Dev strict | Test R@5 | Test strict |
|---|---:|---:|---:|---:|
| Format-SFT v4 | 52.0% | 40.8% | 46.9% | 34.4% |
| KL-fixed GRPO seed42 | 56.1% | 45.9% | 51.6% | 40.6% |
| KL-fixed GRPO seed43 | 56.1% | 45.9% | 48.4% | 37.5% |
| KL-fixed GRPO two-seed mean | 56.1% | 45.9% | 50.0% | 39.1% |

The corrected two-seed mean improves strict success over Format-SFT by +5.1 dev
points and +4.7 test points. This is a modest small-sample main result, not a
large-scale statistically settled result.

Fixed-reader control, where all KL-fixed queries are answered by the same
Format-SFT reader, gives the same two-seed strict result: dev 45.9% and test
39.1%. This means the corrected gain is not explained by reader drift.

The retrieval setting is a frozen Pix2Fact evidence-note BM25 corpus, not live
web search. The dev/test evaluation corpus has 201 short documents with median
length 6 whitespace tokens; see `results/reports/CORPUS_REALITY_AUDIT.md`.

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

- Current v6 status and claim boundary: `docs/CURRENT_RESEARCH_STATUS_2026-07-20.md`
- Current top-conference roadmap: `v6/reports/DAGIG_V6_TOP_CONFERENCE_RESULT_ROADMAP.md`
- v6 code and audit snapshot: `v6/README.md`
- Corrected KL-fixed status: `results/reports/KLFIXED_GRPO_60_REPORT.md`
- Main status: `results/reports/PAPER_MAIN_V1_CURRENT_STATUS.md`
- Audit fixes: `results/reports/CORE_FIX_VALIDATION.md`
- Checker-v4 rescore: `results/reports/CHECKER_V4_RESCORING_REPORT.md`
- Dev-selection correction: `results/reports/PAPER_MAIN_V1_CORRECTED_V4_DEV_SELECTION.md`
- Corpus reality audit: `results/reports/CORPUS_REALITY_AUDIT.md`
- Historical consolidated results: `results/reports/PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`
- Reward audit: `results/reports/REWARD_AUDIT_REPORT.md`
- Seed confirmation: `results/reports/SEED_CONFIRMATION_REPORT.md`
- Goldfixed control: `results/reports/GOLDFIXED_GRPO_60_REPORT.md`
- Evidence chain: `results/reports/MAINLINE_EVIDENCE_CHAIN.md`
- Reproduction appendix: `docs/REPRODUCIBILITY_APPENDIX.md`
- Commands: `scripts/reproduce_main_commands.sh`

## Quick Verification

To verify that the exported paper-main metrics are internally consistent:

```bash
python scripts/verify_paper_main_results.py
```

This checks the corrected KL-fixed summary, fixed-reader control, the main result table, and the seed42/seed43 training-health records. It does not rerun model inference.

## Reproduction Notes

The command template in `scripts/reproduce_main_commands.sh` assumes the original local asset layout from the experiment machine. For a fresh machine, update:

- base model path;
- Format-SFT adapter path;
- train/dev/test JSONL paths;
- BM25 train/eval corpus paths;
- output root.

The core training/evaluation scripts are included, but this repo does not bundle the full dataset or checkpoints.

## Main Claim Boundary

The intended main method is **exact node-level DAG-IG credit over a complete
counterfactual multimodal search DAG**. The current primary experiment is the
direct posterior selector; GRPO/GDPO distillation is deferred until that
selector passes under corrected semantic-support labels.

Do not present DAG-SFT as the main method. Do not use 7B/external-baseline work to support the current 3B paper-main claim. Do not report the old-KL result as paper-facing.
