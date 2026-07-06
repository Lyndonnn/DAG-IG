# DAG-IG Claims And Evidence Matrix

## Purpose

This matrix ties each paper-facing claim to the exact evidence artifact that supports it. It should be used when editing the abstract, introduction, results, and conclusion so that the paper does not overclaim beyond the current experiments.

## Claim Matrix

| ID | Paper-facing claim | Supported wording | Evidence | Do not say |
|---|---|---|---|---|
| C1 | DAG-IG is the main method, not DAG-SFT. | "The paper-main method is node-level DAG-IG credit optimized with grouped GRPO over a two-stage multimodal search agent." | `PAPER_MAIN_EVIDENCE_BRIEF.md`; `PAPER_DRAFT_V0.md`; `latex/main.tex` | "DAG-SFT is the main method" |
| C2 | The rollout schema is visual/query/evidence/answer. | "We model the agent rollout as image+question -> visual_observation -> search_query -> retrieved top-k evidence -> final_answer." | `latex/main.tex`; `figures/dagig_method_diagram.tex`; `latex/algorithm_dagig_grpo.tex` | "The model directly learns region-RL/crop policy end-to-end" |
| C3 | DAG-IG improves Format-SFT in the frozen offline setup. | "DAG-IG seed42 improves strict success from 42.9% to 49.0% on dev and from 34.4% to 40.6% on test." | `PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`; `main_results_table.csv`; `main_results_table.tex` | "DAG-IG solves Pix2Fact" or "large-scale SOTA" |
| C4 | The improvement is confirmed by a second seed. | "Seed43 confirms the improvement direction with 49.0% dev strict and 39.1% test strict." | `SEED_CONFIRMATION_REPORT.md`; `PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md` | "All seeds have identical gains" |
| C5 | Goldfixed is a control, not the promoted main run. | "The goldfixed run verifies reward health after train-corpus label repair, but seed42 remains the main checkpoint because goldfixed has lower test strict." | `GOLDFIXED_GRPO_60_REPORT.md`; `PAPER_MAIN_V1_CURRENT_STATUS.md` | "Goldfixed is the best final model" |
| C6 | The reward is non-collapsed and discriminative. | "Reward AUC is 1.000 for retrieval hit and 0.974 for strict success in seed42; constant groups are 2/240." | `NODE_CREDIT_COMPONENT_ANALYSIS.md`; `node_credit_diagnostic_table.csv` | "The reward is perfect" |
| C7 | Query/evidence/answer components align with their intended outcomes. | "Query and evidence components have AUC(hit)=1.000; answer component has AUC(strict)=1.000 in the seed42 reward analysis." | `NODE_CREDIT_COMPONENT_ANALYSIS.md`; `node_credit_component_summary.json` | "Every component is equally important" |
| C8 | Format/leakage are not the main source of improvement. | "Format credit is low-variance and answer-in-query leakage is near zero; the gains are driven by retrieval/answer paths." | `NODE_CREDIT_COMPONENT_ANALYSIS.md`; `PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md` | "The method only learns JSON format" |
| C9 | Remaining bottlenecks are retrieval misses and reader errors. | "For seed42, dev/test retrieval misses are 42/31 and hit-answer-wrong cases are 8/7." | `PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`; `CASE_STUDY_SUMMARY.md` | "Answer extraction is solved" |
| C10 | Evaluation is offline BM25, not live web search. | "All reported results use a frozen offline BM25 corpus." | `REPRODUCIBILITY_APPENDIX.md`; `PAPER_MAIN_EVIDENCE_BRIEF.md`; `latex/main.tex` | "DAG-IG generalizes to live web search" |
| C11 | Diagnostic branches should stay in appendix. | "DAG-SFT, query reranking, multi-query fusion, and answer repair are diagnostics that motivated the final method." | `APPENDIX_DIAGNOSTIC_RESULTS_INDEX.md`; `latex/appendix.tex` | "DPO/fusion/repair are final paper wins" |
| C12 | The paper package is internally consistent. | "The paper asset audit passes: paths, tables, citations, command syntax, bundle, and counts are consistent." | `PAPER_ASSET_AUDIT_REPORT.md`; `paper_asset_audit.json` | "The target-venue PDF is final" |
| C13 | The generic article-format PDF compiles and renders the required content. | "The generic source bundle compiles locally, and the rendered PDF passes a post-compile audit for the main results, reward diagnostics, rollout schema, limitations, and references." | `PDF_BUILD_PREFLIGHT_REPORT.md`; `POST_COMPILE_PDF_AUDIT.md`; `submission_bundle/main.pdf` | "The venue-template conversion is complete" |

## Main Numeric Claims

| Quantity | Value | Evidence |
|---|---:|---|
| train/dev/test split | 458 / 98 / 64 | `REPRODUCIBILITY_APPENDIX.md`; `derived_manifest.json` |
| eval corpus docs | 201 | `REPRODUCIBILITY_APPENDIX.md`; `paper_asset_audit.json` |
| train corpus docs | 610 | `REPRODUCIBILITY_APPENDIX.md`; `paper_asset_audit.json` |
| Format-SFT dev/test strict | 42.9% / 34.4% | `main_results_table.csv` |
| DAG-IG seed42 dev/test strict | 49.0% / 40.6% | `main_results_table.csv` |
| DAG-IG seed42 dev/test R@5 | 57.1% / 51.6% | `main_results_table.csv` |
| seed42 strict-only wins over Format-SFT | 8 dev / 5 test | `PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`; `CASE_STUDY_SUMMARY.md` |
| seed42 strict-only losses vs Format-SFT | 2 dev / 1 test | `PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`; `CASE_STUDY_SUMMARY.md` |
| seed42 reward AUC hit/strict | 1.000 / 0.974 | `node_credit_diagnostic_table.csv` |
| seed42 constant reward groups | 2 / 240 | `node_credit_diagnostic_table.csv`; `grpo_train_summary.json` |

## Allowed Abstract-Level Claim

Recommended:

> In a frozen offline BM25 Pix2Fact setting, DAG-IG improves strict success over a Format-SFT two-stage baseline from 42.9% to 49.0% on dev and from 34.4% to 40.6% on test. A second seed confirms the recipe, and reward audits show the node-level reward is non-collapsed and predictive of retrieval and strict success.

Avoid:

> DAG-IG solves multimodal web search.

## Completion Note

This matrix is evidence for writing discipline, not additional experimental evidence. If any claim is edited beyond the allowed wording, update this matrix or add stronger evidence before treating the paper as ready.
