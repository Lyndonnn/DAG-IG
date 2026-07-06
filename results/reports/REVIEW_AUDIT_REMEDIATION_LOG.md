# Reviewer Audit Remediation Log

This file maps the external audit findings to the corrected release artifacts. It is the quickest entry point for checking what changed after the audit.

## A. Blocking Issues

| audit item | status | remediation | evidence |
|---|---|---|---|
| A1 wrong KL penalty | fixed | GRPO now uses a non-negative k3 KL estimator rather than a signed policy-reference log-ratio penalty. | `scripts/dagig_grpo/02_train_grpo.py`; `results/reports/CORE_FIX_VALIDATION.md`; `results/reports/KLFIXED_GRPO_60_REPORT.md` |
| A2 test-set selection contamination | fixed | Corrected headline reports KL-fixed seed42/seed43 two-seed mean, not the best test seed. | `results/reports/KLFIXED_GRPO_60_REPORT.md`; `results/tables/main_results_table.csv`; `paper/latex/main.tex` |
| A3 answer checker false positives | fixed | Checker v4 blocks bare AM/PM fallback, substring-boundary false positives, and numeric-range-as-single-answer false positives. | `scripts/dagig_grpo/grpo_utils.py`; `results/reports/CHECKER_V4_RESCORING_REPORT.md`; `results/reports/CORE_FIX_VALIDATION.md` |
| A4 fixed-reader wording mismatch | fixed | Own-reader is the main setting; fixed Format-SFT reader is reported as a control and preserves the KL-fixed two-seed strict result. | `results/reports/KLFIXED_GRPO_60_REPORT.md`; `results/metrics/klfixed_grpo_60_summary.json`; `paper/latex/main.tex` |
| A5 counterfactual/IG overclaim | fixed in wording | Paper-facing text now says node-level credit/reward. Causal counterfactual intervention claims are explicitly disallowed unless added as future experiments. | `README.md`; `paper/latex/main.tex`; `results/reports/CLAIMS_EVIDENCE_MATRIX.md` |
| A6 repo not runnable / hard-coded model path | fixed | Default model is public `Qwen/Qwen2.5-VL-3B-Instruct`; local override uses `DAGIG_LOCAL_3B_MODEL`. Reproduction commands no longer hard-code `/root` model snapshots. | `scripts/dagig_grpo/grpo_utils.py`; `scripts/reproduce_main_commands.sh`; `docs/REPRODUCIBILITY_APPENDIX.md` |

## B. Data And Corpus Issues

| audit item | status | remediation | evidence |
|---|---|---|---|
| Corpus was short evidence notes, not live/noisy web documents | fixed in wording and audit | Corpus reality audit reports 201 dev/test docs, median 6 tokens, and describes the setting as frozen Pix2Fact evidence-note BM25 retrieval. | `results/reports/CORPUS_REALITY_AUDIT.md`; `paper/latex/main.tex`; `docs/REPRODUCIBILITY_APPENDIX.md` |
| Old unclean/oracle sources | guarded | Training utilities reject forbidden model/source names and paper text states old unclean/oracle trajectories are outside the paper-main path. | `scripts/dagig_grpo/grpo_utils.py`; `docs/PAPER_MAIN_V1_SCHEMA.md`; `paper/latex/main.tex` |

## C. Statistical And Selection Claims

| audit item | status | remediation | evidence |
|---|---|---|---|
| Need multi-seed / no test picking | fixed for current package | Corrected report uses seed42/seed43 mean and includes paired comparisons. | `results/reports/KLFIXED_GRPO_60_REPORT.md` |
| Significance should be modest | fixed in wording | Paper text states paired tests are directionally positive but not conventionally significant. | `paper/latex/main.tex`; `docs/REVIEWER_RISK_REGISTER.md`; `results/reports/CLAIMS_EVIDENCE_MATRIX.md` |

## D. Minor Protocol Clarity

| audit item | status | remediation | evidence |
|---|---|---|---|
| Reward/KL/checker version clarity | fixed | Current reports identify k3 KL, checker v4, stage1 loss, 60 steps, 4 generations, top-5 retrieval, and first 320 train samples. | `docs/REPRODUCIBILITY_APPENDIX.md`; `results/reports/KLFIXED_GRPO_60_REPORT.md` |
| Historical reports could be mistaken for current results | fixed | Old-KL, seed-confirmation, goldfixed, augmented-query, and consolidated reports are marked superseded/diagnostic where needed. | `results/reports/PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`; `results/reports/SEED_CONFIRMATION_REPORT.md`; `results/reports/GOLDFIXED_GRPO_60_REPORT.md`; `results/reports/AUGQUERY_GRPO_30_REPORT.md` |

## Verification Commands

Run from the release repo root:

```bash
python scripts/verify_paper_main_results.py
bash -n scripts/reproduce_main_commands.sh
python -m py_compile scripts/dagig_grpo/02_train_grpo.py scripts/dagig_grpo/grpo_utils.py scripts/dagig_grpo/04_summarize_results.py scripts/verify_paper_main_results.py
```

Expected verifier headline:

```text
Corrected KL-fixed paper-main verification passed.
Two-seed KL-fixed strict gain over Format-SFT: dev +5.1, test +4.7.
Core fixes passed: k3 KL, checker v4, training health, fixed reader, and corpus boundary.
```

## Current Paper-Facing Claim

The corrected, conservative claim is:

> In a frozen offline Pix2Fact evidence-note BM25 setting, KL-fixed DAG-IG improves strict success over a Format-SFT two-stage baseline from 40.8% to 45.9% on dev and from 34.4% to 39.1% on test as a two-seed mean. Fixed-reader controls preserve the gain. The result is modest, small-sample, and should be described as node-level credit/reward optimization rather than causal counterfactual intervention.
