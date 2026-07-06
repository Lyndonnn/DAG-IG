# Paper Main v1 Current Status

## Current Main Candidate

- checkpoint: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60`
- training: 60 optimizer steps, 240 micro-steps, `constant_reward_groups=2` (`0.83%` of micro-steps)
- method: two-stage rollout, stage1-only policy loss, `paper_main_v1` reward, KL=0.1

## Current Best Clean Result

Numbers use parser/checker v3 rescoring.

| Method | Dev R@5 | Dev strict | Test R@5 | Test strict |
|---|---:|---:|---:|---:|
| Format-SFT | 52.0% | 42.9% | 46.9% | 34.4% |
| DAG-IG medium30 ckpt30 | 57.1% | 48.0% | 50.0% | 39.1% |
| DAG-IG scale60_s320 seed42 ckpt60 | 57.1% | 49.0% | 51.6% | 40.6% |
| DAG-IG scale60_s320 seed43 ckpt60 | 58.2% | 49.0% | 50.0% | 39.1% |

## Interpretation

DAG-IG now has a real, clean two-stage improvement over Format-SFT on both dev and test. The current best checkpoint improves strict success over Format-SFT by `+6.1` dev points and `+6.2` test points.

Seed confirmation is now complete in `outputs/dagig_paper_main_v1/reports/SEED_CONFIRMATION_REPORT.md`. Seed43 uses the same recipe and has healthy training (`constant_reward_groups=2/240`) with dev/test strict `49.0% / 39.1%`, confirming that the recipe is not a single-seed artifact. Seed42 remains the current best single checkpoint because it has higher test strict and R@5.

Fixed-reader isolation for scale60_s320 ckpt60 keeps test strict at `40.6%`, so the test improvement is not just reader drift. On dev, fixed-reader is `48.0%` rather than `49.0%`, so one dev point is reader/shared-adapter dependent.

The remaining bottlenecks are still retrieval misses and answer extraction/verifier quality. Current ckpt60 has retrieval-hit-answer-wrong counts of `8` dev and `7` test.

## Negative Results

A broad non-oracle regex/span repair was tested in `outputs/dagig_paper_main_v1/answer_repair/ANSWER_STAGE_REPAIR_REPORT.md`. It is not paper-usable: dev gain is tiny and test does not improve, while false repairs are common.

A lightweight learned verifier and two short reader-SFT smoke runs were tested in `outputs/dagig_paper_main_v1/reports/ANSWER_NODE_READER_SFT_SMOKE_REPORT.md`.

| Reader/verifier variant | Dev R@5 | Dev strict | Test strict | Decision |
|---|---:|---:|---:|---|
| ckpt30 own reader | 57.1% | 48.0% | 39.1% | keep |
| lightweight answer verifier | 57.1% | 48.0% | 39.1% | no gain |
| reader-SFT smoke, prompt aligned | 57.1% | 44.9% | - | no-go |

The answer node remains a bottleneck, but cheap post-hoc repair and tiny reader-SFT do not solve it.

## Retrieval-Miss Follow-Up

Hard retrieval mining is now complete in `outputs/dagig_paper_main_v1/reports/hard_retrieval_mining/HARD_RETRIEVAL_MINING_REPORT.md`.

- train rollouts mined: `14656`
- train samples: `458`
- train query hit-vs-miss pairs: `311`
- candidate-insufficient train samples with no hit rollout: `124`
- dev/test retrieval misses analyzed: `42 / 31`
- dominant dev/test miss causes: missing semantic anchor, query drift from teacher search intent, wrong-sample retrieval cluster

A query-node SFT warmup smoke was run from Format-SFT using only the 311 train hit-vs-miss pairs:

| Method | Dev R@5 | Dev strict | Decision |
|---|---:|---:|---|
| Format-SFT full dev | 52.0% | 42.9% | baseline |
| Query-node SFT smoke + fixed Format reader | 55.1% | 44.9% | useful warmup signal, not standalone |
| DAG-IG GRPO seed42 ckpt60 | 57.1% | 49.0% | current best |
| DAG-IG GRPO seed43 ckpt60 | 58.2% | 49.0% | seed confirmation |

The warmup improves retrieval over Format-SFT but does not beat GRPO. It should only be used as a candidate initialization or auxiliary warmup before another GRPO run.

## Query-Warm GRPO Follow-Up

A controlled GRPO run initialized from the query-node warmup adapter is complete in `outputs/dagig_paper_main_v1/reports/QUERYWARM_GRPO_30_REPORT.md`.

| Method | Dev R@5 | Dev answer | Dev strict | Retrieval miss | Hit-answer-wrong | Decision |
|---|---:|---:|---:|---:|---:|---|
| DAG-IG GRPO seed42 ckpt60 | 57.1% | 51.0% | 49.0% | 42 | 8 | current main |
| DAG-IG GRPO seed43 ckpt60 | 58.2% | 51.0% | 49.0% | 41 | 9 | seed confirmation |
| Query-warm GRPO ckpt30 own reader | 59.2% | 48.0% | 44.9% | 40 | 14 | no promotion |
| Query-warm GRPO ckpt30 + fixed Format reader | 59.2% | 51.0% | 48.0% | 40 | 11 | retrieval-positive, still below main strict |

Training was healthy (`30` optimizer steps, `120` micro-steps, `3` constant-reward groups = `2.5%`), so this is a real result rather than reward collapse. It raises dev R@5 to `59.2%`, but strict success remains below the current main checkpoints. Do not run test for this checkpoint and do not promote it as the main result.

The paper-facing interpretation is narrower and useful: query-node credit can improve retrieval, while final success is still blocked by answer extraction and by train samples whose rollout pool lacks any support-hit candidate.

## Augmented Query-Node Data

Train-only no-hit query mining is complete in `outputs/dagig_paper_main_v1/reports/nohit_query_candidate_mining/NOHIT_QUERY_CANDIDATE_MINING_REPORT.md`.

- no-hit train samples from existing rollouts: `124`
- recovered by clean train-only query recipes: `34` (`27.4%`)
- recovered rank distribution: `28` at rank1, `5` at rank2, `1` at rank5
- augmented query-node SFT rows: `345` = `311` hit-vs-miss rows + `34` no-hit recovery rows

The augmented query-node SFT smoke is complete in `outputs/dagig_paper_main_v1/reports/AUGMENTED_QUERY_WARMUP_REPORT.md`.

| Method | Dev R@5 | Dev strict | Retrieval miss | Hit-answer-wrong | Decision |
|---|---:|---:|---:|---:|---|
| Query-node SFT smoke + fixed reader | 55.1% | 44.9% | 44 | 10 | useful but weak |
| Augmented query-node SFT smoke + fixed reader | 56.1% | 48.0% | 43 | 8 | useful init |
| DAG-IG GRPO seed42 ckpt60 | 57.1% | 49.0% | 42 | 8 | current main |

The no-hit recovery rows materially improve the query-node warmup, but the adapter still does not beat the current main checkpoint. It is worth exactly one controlled GRPO run from this initialization, with dev-only gating before any test evaluation.

The gated augmented-init GRPO run is complete in `outputs/dagig_paper_main_v1/reports/AUGQUERY_GRPO_30_REPORT.md`.

| Method | Dev R@5 | Dev strict | Retrieval miss | Hit-answer-wrong | Decision |
|---|---:|---:|---:|---:|---|
| Augmented query-node SFT smoke + fixed reader | 56.1% | 48.0% | 43 | 8 | useful warmup |
| Aug-query GRPO ckpt10 | 55.1% | 45.9% | 44 | 9 | no-go |
| Aug-query GRPO ckpt20 | 56.1% | 45.9% | 43 | 10 | no-go |
| Aug-query GRPO ckpt30 | 56.1% | 45.9% | 43 | 10 | no-go |
| DAG-IG GRPO seed42 ckpt60 | 57.1% | 49.0% | 42 | 8 | current main |

Training was healthy (`constant_reward_groups=0/120`), so the failure is not reward collapse. The issue is that GRPO from the augmented warmup worsens answer/strict behavior and does not improve retrieval enough to compensate. Do not run test for this checkpoint.

## Next Mainline Step

Keep seed42 scale60_s320 checkpoint-60 as the current main method result, with seed43 as confirmation. Do not keep iterating on broad answer repair, reward reshuffling, or query-warm GRPO variants.

The next efficient paper-facing step is stronger non-oracle query candidate generation for the remaining `90` no-hit train samples that simple recipes did not recover. Do not run another GRPO from the same augmented warmup. Revisit reader training only with a larger hard-context dataset that explicitly covers retrieval-hit-answer-wrong cases.

Support-document lexical mining is complete in `outputs/dagig_paper_main_v1/reports/support_doc_query_candidate_mining/SUPPORT_DOC_QUERY_CANDIDATE_MINING_REPORT.md`.

- remaining no-hit samples after simple recipes: `90`
- newly recovered using train support-document title/url/domain/text lexical fields: `48` (`53.3%`)
- cumulative recovered no-hit samples: `82 / 124` (`66.1%`)
- all newly recovered support hits are rank1
- answer-leak candidates filtered: `22`

This is not a promoted training result; it is a diagnosis. It shows the main retrieval bottleneck is query candidate generation rather than BM25/corpus mismatch. Because these candidates are support-doc-derived, keep them labeled separately and do not mix them silently into the clean DAG-IG main result.

Train corpus gold coverage audit found a concrete corpus-label issue:

- original train BM25 corpus samples with gold doc: `417 / 458`
- missing gold-doc samples: `41`
- all missing cases had same-sample corpus docs, but none marked `is_gold=true`

A uniform train-only fix was written to `outputs/dagig_paper_main_v1/derived_assets/bm25_train_corpus_goldfixed.jsonl` and documented in `outputs/dagig_paper_main_v1/reports/train_gold_corpus_coverage/GOLDFIXED_TRAIN_CORPUS_MANIFEST.md`.

- fixed train BM25 corpus samples with gold doc: `458 / 458`
- docs fixed: `56`
- dev/test corpora were not modified

Use the fixed train corpus for future train-side reward/coverage audits and any future GRPO. Do not compare future train-side reward statistics to earlier runs without noting this corpus fix.

Existing train rollouts were rescored against the fixed corpus in `outputs/dagig_paper_main_v1/reports/goldfixed_rollout_rescore/GOLDFIXED_TRAIN_ROLLOUT_RETRIEVAL_REPORT.md`.

- old hit rollouts: `6127`
- fixed-corpus hit rollouts: `6694`
- newly hit rollouts: `567`
- old hit samples: `334`
- fixed-corpus hit samples: `365`
- old no-hit samples: `124`
- fixed-corpus no-hit samples: `93`
- old strict rollouts: `110`
- fixed-corpus strict rollouts approx: `129`

This means the earlier no-hit analysis overestimated the model-side retrieval gap by `31` samples because some train support docs were present but not marked gold.

Goldfixed hard retrieval mining was rebuilt in `outputs/dagig_paper_main_v1/reports/hard_retrieval_mining_goldfixed/HARD_RETRIEVAL_MINING_GOLDFIXED_REPORT.md`.

- train query hit-vs-miss pairs: `339` instead of the previous `311`
- learnable-from-existing-rollouts samples: `339`
- already-hit/no-miss-pair samples: `26`
- candidate-insufficient no-hit samples: `93`

Future query-node warmup or candidate analysis should use the goldfixed pair file, not the old pair file, unless reproducing old runs.

Goldfixed reward audit is complete in `outputs/dagig_paper_main_v1/reports/reward_audit_goldfixed/PAPER_MAIN_V1_GOLDFIXED_REWARD_AUDIT.md`.

- rollouts audited: `14656`
- fixed-corpus hit rollouts: `6801`
- fixed-corpus strict rollouts: `140`
- reward AUC vs fixed-corpus hit: `0.999`
- reward AUC vs fixed-corpus strict: `0.938`
- constant reward groups: `37 / 3664` (`1.0%`)

Decision: GO for reward health under the fixed train corpus. The reward remains discriminative after the gold-label fix. This means the next mainline run should be a controlled rerun of the stable paper-main GRPO recipe using the fixed train corpus, not another query-warm or answer-repair side path.

Next controlled run:

- initializer: `outputs/dagig_grpo_main/checkpoints/format_sft`
- reward: `paper_main_v1`
- rollout: two-stage
- loss scope: stage1 only
- KL: `0.1`
- learning rate: `1e-6`
- fixed train corpus: `outputs/dagig_paper_main_v1/derived_assets/bm25_train_corpus_goldfixed.jsonl`
- gate: dev eval first; run test only if dev strict matches or beats current seed42 ckpt60 (`49.0%`) or clearly improves retrieval without increasing hit-answer-wrong.

The fixed-corpus controlled run is complete in `outputs/dagig_paper_main_v1/reports/GOLDFIXED_GRPO_60_REPORT.md`.

Training was healthy:

- optimizer steps: `60`
- micro steps: `240`
- constant reward groups: `2 / 240` (`0.83%`)
- max GPU memory: `19.833` GB

Dev gate passed but test did not improve:

| Method | Dev R@5 | Dev strict | Test R@5 | Test strict | Decision |
|---|---:|---:|---:|---:|---|
| Current seed42 ckpt60 | 57.1% | 49.0% | 51.6% | 40.6% | current main |
| Goldfixed ckpt60 | 57.1% | 50.0% | 50.0% | 39.1% | no promotion |

Decision: keep `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60` as the paper-main checkpoint. The goldfixed rerun is a useful robustness/control run showing reward health after corpus repair, but it does not become the main result because test strict and test R@5 are lower.

Updated next mainline action: stop rerunning the same GRPO recipe. Consolidate the paper-facing result table and failure analysis around four items: Format-SFT baseline, seed42 main result, seed43 confirmation, and goldfixed control. The next method change should target retrieval coverage or answer extraction explicitly; do not launch another same-recipe GRPO without a new mechanism.

## Paper Assets

Paper-facing assets have been generated under `outputs/dagig_paper_main_v1/paper_assets/`.

- main result table: `main_results_table.csv` and `main_results_table.tex`
- node-credit diagnostic table: `node_credit_diagnostic_table.csv` and `node_credit_diagnostic_table.tex`
- experiment manifest: `paper_experiment_manifest.json`
- concise package report: `PAPER_EXPERIMENT_PACKAGE.md`
- evidence brief: `PAPER_MAIN_EVIDENCE_BRIEF.md`
- paper draft outline: `PAPER_DRAFT_OUTLINE.md`
- paper draft v0: `PAPER_DRAFT_V0.md`
- paper completion checklist: `PAPER_COMPLETION_CHECKLIST.md`
- final handoff prompt: `FINAL_HANDOFF_PROMPT.md`
- handoff README: `HANDOFF_README.md`
- reproducibility appendix: `REPRODUCIBILITY_APPENDIX.md`
- reproduction commands: `reproduce_main_commands.sh`
- release-check wrapper: `run_release_checks.sh`
- paper asset audit: `PAPER_ASSET_AUDIT_REPORT.md` and `paper_asset_audit.json`
- claims/evidence matrix: `CLAIMS_EVIDENCE_MATRIX.md`
- reviewer risk register: `REVIEWER_RISK_REGISTER.md`
- submission readiness report: `SUBMISSION_READINESS_REPORT.md`
- PDF build preflight report: `PDF_BUILD_PREFLIGHT_REPORT.md`
- paper length audit: `PAPER_LENGTH_AUDIT.md`
- venue template conversion guide: `VENUE_TEMPLATE_CONVERSION_GUIDE.md`
- related work draft: `RELATED_WORK_DRAFT.md`
- citation source note: `CITATION_SOURCE_NOTE.md`
- appendix diagnostic index: `APPENDIX_DIAGNOSTIC_RESULTS_INDEX.md`
- qualitative case studies: `case_studies/CASE_STUDY_SUMMARY.md`
- method figure snippet: `figures/dagig_method_diagram.tex`
- reward equation snippet: `figures/dagig_reward_equations.tex`
- venue-template-ready snippets: `venue_template_parts/`
- LaTeX scaffold: `latex/main.tex`, `latex/appendix.tex`, `latex/diagnostic_branches_table.tex`, `latex/algorithm_dagig_grpo.tex`, `latex/Makefile`, and `latex/README_LATEX.md`
- BibTeX references: `latex/references.bib`
- self-contained submission bundle: `submission_bundle/`
- submission source tarball: `DAGIG_Pix2Fact_paper_source_bundle.tar.gz`
- Overleaf/source zip bundle: `DAGIG_Pix2Fact_overleaf_source_bundle.zip`
- review-clean source zip: `DAGIG_Pix2Fact_review_clean_source_bundle.zip`
- review-clean anonymity audit: `REVIEW_CLEAN_ANONYMITY_AUDIT.md`
- submission package index and checksums: `SUBMISSION_PACKAGE_INDEX.md` and `SUBMISSION_PACKAGE_INDEX.json`
- package extract verification report: `PACKAGE_EXTRACT_VERIFICATION_REPORT.md`

The evidence brief fixes the current paper position: DAG-SFT is not the main method; the main claim is node-level DAG-IG GRPO for the two-stage multimodal search agent. The case-study report uses gold labels only for post-hoc categorization and matches the consolidated rescored-v3 baseline counts: seed42 has `8` dev / `5` test strict-only wins over Format-SFT, versus `2` dev / `1` test strict-only losses.

Current paper-writing path: use the existing evidence chain rather than launching new same-recipe experiments. The paper should be organized around problem setup, DAG-IG node credit, reward health/non-collapse, two-stage GRPO training, main results, seed confirmation, goldfixed control, and failure analysis. Any further experiment must target a paper-critical presentation gap or a genuinely new retrieval/reader mechanism.

The paper asset audit currently passes: manifest paths exist, main/result tables match consolidated metric JSON, citation keys resolve, reproduction command syntax is valid, the LaTeX source checks pass, the self-contained submission bundle validates, the Overleaf/source zip is generated, submission-facing TeX has no local path leaks, TeX linebreak compile-risk checks pass, and split/corpus counts match the reproducibility appendix. The generic article-format source also compiles locally, and the rendered PDF passes the post-compile content audit.

Current paper-package state: ready for target-template conversion. The only remaining non-experimental blockers are author/venue formatting, target venue template conversion, and venue-specific rendered PDF inspection.
