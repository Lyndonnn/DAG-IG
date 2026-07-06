# Core Repository Manifest

This manifest records what was exported from the full experiment workspace into this clean GitHub-ready repository.

## Included

### Code

- `scripts/dagig_grpo/`
  - derived asset builder
  - GRPO trainer
  - GRPO evaluator
  - two-stage evaluator
  - reward/evaluation utilities
- `scripts/dagig_paper_main/`
  - paper-main protocol construction
  - reward audits
  - result consolidation
  - query/error diagnostics
  - paper asset builders
- `scripts/reproduce_main_commands.sh`

### Results

- `results/reports/PAPER_MAIN_V1_CURRENT_STATUS.md`
- `results/reports/PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`
- `results/reports/PAPER_MAIN_V1_PROGRESS_REPORT.md`
- `results/reports/REWARD_AUDIT_REPORT.md`
- `results/reports/SEED_CONFIRMATION_REPORT.md`
- `results/reports/GOLDFIXED_GRPO_60_REPORT.md`
- selected retrieval, node-credit, and failure-analysis report summaries
- consolidated JSON metrics in `results/metrics/`
- paper table sources in `results/tables/`

### Paper Assets

- LaTeX scaffold in `paper/latex/`
- method/reward TeX snippets in `paper/figures/`
- paper table TeX in `paper/tables/`
- venue-template-ready body fragments in `paper/venue_template_parts/`

## Excluded

- `outputs/**/checkpoints/`
- `adapter_model.safetensors`
- raw Pix2Fact images
- raw/train/dev/test data files
- downloaded zip packages
- per-sample prediction JSONL files
- compiled LaTeX outputs
- old 414 diagnostics and non-mainline branches
- 7B exploratory artifacts except for textual guidance in reports

## Why This Export Is Small

The full workspace contains many debugging branches. This repo is meant to preserve the paper-facing code and evidence chain, not every intermediate experiment. The excluded files are either too large for GitHub, not redistributable as plain Git assets, or not part of the final method claim.

## Suggested External Releases

If public release is required, publish these separately:

1. Pix2Fact-derived clean assets, if license allows.
2. Format-SFT adapter.
3. Main DAG-IG seed42 adapter.
4. Seed43 confirmation adapter.
5. Goldfixed-control adapter.

Use Git LFS or Hugging Face for adapter weights.
