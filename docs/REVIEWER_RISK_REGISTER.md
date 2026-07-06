# Reviewer Risk Register

## Purpose

This register lists likely reviewer objections and the current evidence or wording that should answer them. It is meant to keep the paper focused and prevent late-stage overclaiming.

## High-Priority Risks

### R1. "The gains are modest."

Risk level: high.

Answer:

- Acknowledge the gains are modest.
- Emphasize consistency across dev/test and seed confirmation.
- Frame the contribution as credit assignment and reward audit, not a large benchmark win.

Evidence:

- seed42 improves strict success by +6.1 dev points and +6.2 test points.
- seed43 confirms the direction: 49.0% dev strict and 39.1% test strict.

Paper wording:

> The gains are modest but consistent, and the reward audit shows the node-level credit signal is meaningful.

Do not say:

> DAG-IG achieves decisive or SOTA performance.

### R2. "Is the reward collapsed or just learning format?"

Risk level: high.

Answer:

- Show reward AUC and constant group rate.
- Show format component is low-variance and not the main driver.

Evidence:

- seed42 reward AUC hit/strict: 1.000 / 0.974.
- constant reward groups: 2/240.
- format component AUC is around 0.5, while query/evidence/answer components align with retrieval/strict outcomes.

Paper location:

- Reward and Credit Diagnostics section.
- Table: `node_credit_diagnostic_table.tex`.

### R3. "Is this really DAG-IG, or just SFT / formatting?"

Risk level: high.

Answer:

- State clearly that DAG-SFT is not the main method.
- Main method is GRPO optimized with node-level DAG-IG reward.
- Format-SFT is the initializer and baseline.

Evidence:

- `PAPER_MAIN_EVIDENCE_BRIEF.md`
- `latex/algorithm_dagig_grpo.tex`
- `NODE_CREDIT_COMPONENT_ANALYSIS.md`

### R4. "Why not DPO / preference tuning / query reranking?"

Risk level: medium.

Answer:

- These were tested as diagnostics.
- Equal-candidate scoring and query reranking did not establish a reliable main path.
- The paper therefore uses node-level reward with GRPO as the main validation.

Evidence:

- `APPENDIX_DIAGNOSTIC_RESULTS_INDEX.md`
- `latex/appendix.tex`

Paper wording:

> Preference and reranking routes exposed useful diagnostics but did not provide a reliable final method under clean non-oracle evaluation.

### R5. "Is the evaluation live web search?"

Risk level: high.

Answer:

- No. Evaluation is frozen offline BM25.
- This is a limitation and also an auditability choice.

Evidence:

- `REPRODUCIBILITY_APPENDIX.md`
- `latex/main.tex`, Limitations section.

Do not say:

> DAG-IG works on real web search.

### R6. "Does goldfixed change the main result?"

Risk level: medium.

Answer:

- No. Goldfixed is a robustness/control run after a train-corpus label repair.
- It improves dev but not test; seed42 remains the main checkpoint.

Evidence:

- goldfixed dev/test strict: 50.0% / 39.1%.
- seed42 dev/test strict: 49.0% / 40.6%.

### R7. "Are failures mostly answer extraction?"

Risk level: medium.

Answer:

- Failures are mixed, but retrieval misses remain the largest count.
- Reader errors remain visible when retrieval succeeds.

Evidence:

- seed42 retrieval misses: 42 dev / 31 test.
- seed42 hit-answer-wrong: 8 dev / 7 test.

Paper wording:

> Remaining errors are dominated by retrieval misses, with additional reader failures after successful retrieval.

### R8. "Is the paper package reproducible?"

Risk level: medium.

Answer:

- Provide reproducibility appendix and command templates.
- Provide self-contained LaTeX source bundle.
- Provide paper asset audit.

Evidence:

- `REPRODUCIBILITY_APPENDIX.md`
- `reproduce_main_commands.sh`
- `submission_bundle/`
- `PAPER_ASSET_AUDIT_REPORT.md`
- `MAINLINE_EVIDENCE_CHAIN.md`
- `TEXT_FINALIZATION_AUDIT.md`
- `VENUE_WORKSPACE_AUDIT.md`
- `SHA256SUMS.txt`

Current status:

- Generic source PDF compilation has been audited in this environment with `pdflatex`/`bibtex`.
- The rendered PDF passes content, layout, and text-finalization audits.
- Source/review-clean bundles and checksum verification pass. The remaining reproducibility caveat is venue-specific formatting, not the experimental pipeline.

## Medium-Priority Risks

### R9. "The dataset is small."

Answer:

- Acknowledge split size.
- Position the work as a controlled diagnostic of credit assignment.
- Do not overclaim generality.

Evidence:

- train/dev/test = 458 / 98 / 64.

### R10. "Could answer leakage explain the result?"

Answer:

- The eval tracks answer-in-query leakage.
- Main runs have zero or near-zero answer-in-query.

Evidence:

- `PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`
- seed42 answer-in-query: dev 1.0%, test 0.0%.

### R11. "Does the reader change make comparison unfair?"

Answer:

- Main evaluation uses each model's own adapter as reader, and fixed-reader isolation was tested earlier.
- The paper should emphasize the two-stage setup and note fixed-reader diagnostics if needed.

Evidence:

- `PAPER_MAIN_V1_CURRENT_STATUS.md`
- fixed-reader notes for scale60_s320.

## Final Writing Guidance

The safest paper stance is:

1. DAG-IG is a credit-assignment method for multimodal search agents.
2. The current implementation optimizes stage-1 visual/query behavior.
3. The offline Pix2Fact result shows consistent, modest gains over Format-SFT.
4. Reward audits prove the credit signal is not collapsed.
5. Remaining work is retrieval coverage and reader extraction.

Do not let the paper drift into claims about live web search, solved answer extraction, or final DPO/RL beyond the current GRPO evidence.
