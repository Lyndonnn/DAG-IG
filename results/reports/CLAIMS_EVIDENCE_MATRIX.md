# DAG-IG Claims And Evidence Matrix

## Purpose

This matrix ties each paper-facing claim to the exact evidence artifact that supports it. It should be used when editing the abstract, introduction, results, and conclusion so that the paper does not overclaim beyond the current experiments.

## Claim Matrix

| ID | Paper-facing claim | Supported wording | Evidence | Do not say |
|---|---|---|---|---|
| C1 | DAG-IG is the main method, not DAG-SFT. | "The paper-main method is node-level DAG-IG credit optimized with grouped GRPO over a two-stage multimodal search agent." | `PAPER_MAIN_EVIDENCE_BRIEF.md`; `PAPER_MAIN_V1_CURRENT_STATUS.md`; `latex/main.tex` | "DAG-SFT is the main method" |
| C2 | The rollout schema is visual/query/evidence/answer. | "We model the agent rollout as image+question -> visual_observation -> search_query -> retrieved top-k evidence -> final_answer." | `latex/main.tex`; `figures/dagig_method_diagram.tex`; `latex/algorithm_dagig_grpo.tex` | "The model directly learns region-RL/crop policy end-to-end" |
| C3 | DAG-IG improves Format-SFT in the frozen offline setup. | "KL-fixed DAG-IG GRPO improves strict success from 40.8% to 45.9% on dev and from 34.4% to 39.1% on test as a two-seed mean." | `KLFIXED_GRPO_60_REPORT.md`; `main_results_table.csv`; `main_results_table.tex`; `klfixed_grpo_60_summary.json` | "DAG-IG solves Pix2Fact" or "large-scale SOTA" |
| C4 | The result is not selected by test performance. | "The corrected headline reports the seed42/seed43 mean rather than selecting the best test seed." | `KLFIXED_GRPO_60_REPORT.md`; `PAPER_MAIN_V1_CURRENT_STATUS.md` | "Seed42 is the main checkpoint because it has the best test score" |
| C5 | Old-KL and goldfixed runs are diagnostics/controls, not the corrected headline. | "Old-KL results are diagnostic only; the corrected paper-facing run uses k3 KL and checker v4." | `KLFIXED_GRPO_60_REPORT.md`; `CORE_FIX_VALIDATION.md`; `PAPER_MAIN_V1_CURRENT_STATUS.md` | "Old-KL seed42 is the final headline" |
| C6 | The corrected reward is non-collapsed in training. | "The KL-fixed reruns have low constant-reward rates: 3/240 for seed42 and 1/240 for seed43." | `KLFIXED_GRPO_60_REPORT.md`; `klfixed_grpo_60_summary.json` | "The reward is perfect" |
| C7 | Query/evidence/answer reward components are diagnostic, not causal proof. | "Node-credit diagnostics show alignment with retrieval and strict outcomes, but do not establish causal counterfactual intervention effects." | `NODE_CREDIT_COMPONENT_ANALYSIS.md`; `CORPUS_REALITY_AUDIT.md`; `latex/main.tex` | "DAG-IG has proven causal counterfactual credit assignment" |
| C8 | Format/leakage are not the main source of improvement. | "Format credit is low-variance and answer-in-query leakage is near zero; the gains are driven by retrieval/answer paths." | `NODE_CREDIT_COMPONENT_ANALYSIS.md`; `PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md` | "The method only learns JSON format" |
| C9 | Remaining bottlenecks are retrieval misses and reader errors. | "For KL-fixed seed42, dev/test retrieval misses are 43/31 and hit-answer-wrong cases are 10/7; for seed43 they are 43/33 and 10/7." | `KLFIXED_GRPO_60_REPORT.md`; `klfixed_grpo_60_summary.json` | "Answer extraction is solved" |
| C10 | Evaluation is offline BM25 over evidence notes, not live web search. | "All reported results use a frozen Pix2Fact evidence-note BM25 corpus; the dev/test corpus has 201 short docs with median length 6 tokens." | `CORPUS_REALITY_AUDIT.md`; `REPRODUCIBILITY_APPENDIX.md`; `latex/main.tex` | "DAG-IG generalizes to live web search or noisy full web pages" |
| C11 | Diagnostic branches should stay in appendix. | "DAG-SFT, query reranking, multi-query fusion, and answer repair are diagnostics that motivated the final method." | `APPENDIX_DIAGNOSTIC_RESULTS_INDEX.md`; `latex/appendix.tex` | "DPO/fusion/repair are final paper wins" |
| C12 | The paper package is internally consistent. | "The paper asset audit passes: paths, tables, citations, command syntax, bundle, and counts are consistent." | `PAPER_ASSET_AUDIT_REPORT.md`; `paper_asset_audit.json` | "The target-venue PDF is final" |
| C13 | Release verification passes for the corrected package. | "The verifier checks corrected KL-fixed metrics, fixed-reader control, checker v4, k3 KL, runnable imports, and corpus-boundary audit." | `scripts/verify_paper_main_results.py`; `CORE_FIX_VALIDATION.md`; `CORPUS_REALITY_AUDIT.md` | "The final venue submission PDF is already complete" |

## Main Numeric Claims

| Quantity | Value | Evidence |
|---|---:|---|
| train/dev/test split | 458 / 98 / 64 | `REPRODUCIBILITY_APPENDIX.md`; `derived_manifest.json` |
| eval corpus docs | 201 | `REPRODUCIBILITY_APPENDIX.md`; `paper_asset_audit.json` |
| train corpus docs | 610 | `REPRODUCIBILITY_APPENDIX.md`; `paper_asset_audit.json` |
| eval corpus median doc length | 6 tokens | `CORPUS_REALITY_AUDIT.md` |
| eval gold-note answer embedded rate | 80.7% | `CORPUS_REALITY_AUDIT.md` |
| dev/test gold-doc coverage upper bound | 93.9% / 90.6% | `CORPUS_REALITY_AUDIT.md`; `corpus_reality_audit.json` |
| Format-SFT v4 dev/test strict | 40.8% / 34.4% | `main_results_table.csv` |
| KL-fixed two-seed mean dev/test strict | 45.9% / 39.1% | `main_results_table.csv`; `klfixed_grpo_60_summary.json` |
| KL-fixed two-seed mean dev/test R@5 | 56.1% / 50.0% | `main_results_table.csv`; `klfixed_grpo_60_summary.json` |
| KL-fixed strict gain over Format-SFT | +5.1 dev / +4.7 test | `KLFIXED_GRPO_60_REPORT.md`; `klfixed_grpo_60_summary.json` |
| KL-fixed seed42 constant reward groups | 3 / 240 | `KLFIXED_GRPO_60_REPORT.md`; `grpo_train_summary.json` |
| KL-fixed seed43 constant reward groups | 1 / 240 | `KLFIXED_GRPO_60_REPORT.md`; `grpo_train_summary.json` |

## Allowed Abstract-Level Claim

Recommended:

> In a frozen offline Pix2Fact evidence-note BM25 setting, KL-fixed DAG-IG improves strict success over a Format-SFT two-stage baseline from 40.8% to 45.9% on dev and from 34.4% to 39.1% on test as a two-seed mean. Fixed-reader controls preserve the gain, and reward audits show low constant-reward rates. The result is modest and small-sample, and does not establish live-web or causal-intervention claims.

Avoid:

> DAG-IG solves multimodal web search.

## Completion Note

This matrix is evidence for writing discipline, not additional experimental evidence. If any claim is edited beyond the allowed wording, update this matrix or add stronger evidence before treating the paper as ready.
