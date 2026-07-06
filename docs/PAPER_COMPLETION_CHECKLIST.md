# DAG-IG Paper Completion Checklist

## Current Status

The paper-main experimental path is frozen, the generic article-format paper source compiles, and the package is ready for target venue conversion.

Completed:

- main checkpoint selected: seed42 `scale60_s320` checkpoint-60;
- seed43 confirmation complete;
- goldfixed control complete;
- node-credit reward audit complete;
- main table and reward diagnostic table generated;
- qualitative case-study table generated;
- markdown paper draft v0 generated;
- verified LaTeX source generated.
- related work draft and BibTeX generated;
- reproducibility appendix and command templates generated.
- paper asset audit generated and passing.
- generic article-format PDF compiled locally and passed post-compile content audit.
- LaTeX appendix, diagnostic branch table, and Makefile source check generated.
- DAG-IG GRPO algorithm box generated and included in the method section.
- Claims/evidence matrix and reviewer risk register generated.
- Handoff README generated and included in the submission bundle.
- Self-contained submission bundle and source tarball generated.
- Overleaf/source zip bundle generated.
- submission-facing TeX path-leak audit generated.
- final handoff prompt generated for the next TeX/venue conversion step.
- mainline evidence chain generated and passing.
- text-finalization audit generated and passing.
- artifact checksum manifest generated and passing.
- venue-conversion workspace generated, generic-compiled, and audited.

Current main claim:

DAG-IG node-level GRPO improves the Format-SFT two-stage Pix2Fact agent under a frozen offline BM25 setting, and the node-level reward is discriminative/non-collapsed.

## Remaining Paper Work

### Required

1. Convert the verified generic source into the target venue template.
2. Replace placeholder author block and venue formatting.
3. Recompile after venue conversion and rerun post-compile content/layout audits, then inspect the rendered text for unresolved placeholders or venue-template residue.
4. Decide how much diagnostic history belongs in appendix under the venue page limit.
5. Tighten the dataset paragraph if the target venue requires more exact packaging details.

### Optional But Useful

1. Add one compressed diagnostic appendix table from `APPENDIX_DIAGNOSTIC_RESULTS_INDEX.md`.
2. Convert the TikZ method diagram to PDF/PNG for easier camera-ready editing.
3. Add one figure showing strict success gain/loss counts.
4. Add one appendix case table with 2 wins and 1 loss.

## Do Not Do Unless A New Mechanism Is Defined

- another same-recipe GRPO run;
- more DAG-SFT trace training;
- DPO from weak preference scorer outputs;
- broad regex/span answer repair;
- query reranking/switching without stronger confidence features;
- multi-query fusion without a clean selector/reader mechanism;
- live web search experiments.

## Paper Evidence Sources

- paper draft v0: `outputs/dagig_paper_main_v1/paper_assets/PAPER_DRAFT_V0.md`
- LaTeX scaffold: `outputs/dagig_paper_main_v1/paper_assets/latex/main.tex`
- main results table: `outputs/dagig_paper_main_v1/paper_assets/main_results_table.tex`
- node-credit table: `outputs/dagig_paper_main_v1/paper_assets/node_credit_diagnostic_table.tex`
- method diagram: `outputs/dagig_paper_main_v1/paper_assets/figures/dagig_method_diagram.tex`
- reward equations: `outputs/dagig_paper_main_v1/paper_assets/figures/dagig_reward_equations.tex`
- case studies: `outputs/dagig_paper_main_v1/paper_assets/case_studies/CASE_STUDY_SUMMARY.md`
- diagnostic index: `outputs/dagig_paper_main_v1/paper_assets/APPENDIX_DIAGNOSTIC_RESULTS_INDEX.md`
- related work draft: `outputs/dagig_paper_main_v1/paper_assets/RELATED_WORK_DRAFT.md`
- references: `outputs/dagig_paper_main_v1/paper_assets/latex/references.bib`
- reproducibility appendix: `outputs/dagig_paper_main_v1/paper_assets/REPRODUCIBILITY_APPENDIX.md`
- reproduction commands: `outputs/dagig_paper_main_v1/paper_assets/reproduce_main_commands.sh`
- paper asset audit: `outputs/dagig_paper_main_v1/paper_assets/PAPER_ASSET_AUDIT_REPORT.md`
- handoff README: `outputs/dagig_paper_main_v1/paper_assets/HANDOFF_README.md`
- claims/evidence matrix: `outputs/dagig_paper_main_v1/paper_assets/CLAIMS_EVIDENCE_MATRIX.md`
- reviewer risk register: `outputs/dagig_paper_main_v1/paper_assets/REVIEWER_RISK_REGISTER.md`
- LaTeX Makefile: `outputs/dagig_paper_main_v1/paper_assets/latex/Makefile`
- LaTeX appendix: `outputs/dagig_paper_main_v1/paper_assets/latex/appendix.tex`
- LaTeX algorithm box: `outputs/dagig_paper_main_v1/paper_assets/latex/algorithm_dagig_grpo.tex`
- submission bundle: `outputs/dagig_paper_main_v1/paper_assets/submission_bundle/`
- submission tarball: `outputs/dagig_paper_main_v1/paper_assets/DAGIG_Pix2Fact_paper_source_bundle.tar.gz`
- Overleaf/source zip: `outputs/dagig_paper_main_v1/paper_assets/DAGIG_Pix2Fact_overleaf_source_bundle.zip`
- final handoff prompt: `outputs/dagig_paper_main_v1/paper_assets/FINAL_HANDOFF_PROMPT.md`
- bundled reproduction commands: `outputs/dagig_paper_main_v1/paper_assets/submission_bundle/scripts/reproduce_main_commands.sh`
- mainline evidence chain: `outputs/dagig_paper_main_v1/paper_assets/MAINLINE_EVIDENCE_CHAIN.md`
- text-finalization audit: `outputs/dagig_paper_main_v1/paper_assets/TEXT_FINALIZATION_AUDIT.md`
- venue workspace audit: `outputs/dagig_paper_main_v1/paper_assets/VENUE_WORKSPACE_AUDIT.md`
- artifact checksums: `outputs/dagig_paper_main_v1/paper_assets/SHA256SUMS.txt`

## Completion Gate

The project should be considered venue-submission complete when:

1. the venue-template source compiles with `make all` or the venue's equivalent build command;
2. all main claims in abstract/introduction/results match `PAPER_MAIN_EVIDENCE_BRIEF.md`;
3. all tables match `main_results_table.tex` and `node_credit_diagnostic_table.tex`;
4. limitations explicitly state offline BM25, modest sample size, and remaining reader/retrieval bottlenecks;
5. no diagnostic branch is presented as the main method.
6. `python scripts/dagig_paper_main/29_audit_paper_assets.py` passes after final template conversion.
7. submission-facing TeX files contain no local `outputs/`, `/root/`, or internal script paths.
8. the venue-rendered PDF passes post-compile content and layout audits.
9. final upload artifacts pass checksum verification.
