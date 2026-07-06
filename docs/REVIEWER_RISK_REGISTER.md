# Reviewer Risk Register

## Purpose

This register lists likely reviewer objections and the current evidence or wording that should answer them. It is meant to keep the paper focused and prevent late-stage overclaiming.

## High-Priority Risks

### R1. "The gains are modest."

Risk level: high.

Answer:

- Acknowledge the gains are modest.
- Emphasize the corrected two-seed mean and fixed-reader control.
- Frame the contribution as credit assignment and reward audit, not a large benchmark win.

Evidence:

- KL-fixed two-seed mean improves strict success by +5.1 dev points and +4.7 test points.
- Individual paired tests are directionally positive but not conventionally significant.

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

- KL-fixed seed42 constant reward groups: 3/240.
- KL-fixed seed43 constant reward groups: 1/240.
- the earlier high constant-reward concern does not apply to the KL-fixed reruns.
- reward-component AUC is diagnostic and should not be framed as causal proof.

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

### R6. "Are old-KL or goldfixed runs the main result?"

Risk level: medium.

Answer:

- No. Old-KL and goldfixed runs are diagnostics/controls.
- The corrected headline is the KL-fixed seed42/seed43 mean under checker v4 and k3 KL.

Evidence:

- KL-fixed two-seed mean dev/test strict: 45.9% / 39.1%.
- Format-SFT v4 dev/test strict: 40.8% / 34.4%.

### R7. "Are failures mostly answer extraction?"

Risk level: medium.

Answer:

- Failures are mixed, but retrieval misses remain the largest count.
- Reader errors remain visible when retrieval succeeds.

Evidence:

- KL-fixed seed42 retrieval misses: 43 dev / 31 test.
- KL-fixed seed42 hit-answer-wrong: 10 dev / 7 test.

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
- KL-fixed main runs have zero or near-zero answer-in-query.

Evidence:

- `PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md`
- seed42 answer-in-query: dev 1.0%, test 0.0%.

### R11. "Does the reader change make comparison unfair?"

Answer:

- Main evaluation uses each model's own adapter as reader.
- Fixed-reader control uses the same Format-SFT reader for all KL-fixed queries and preserves the two-seed strict result, closing the main reader-drift concern.

Evidence:

- `PAPER_MAIN_V1_CURRENT_STATUS.md`
- `KLFIXED_GRPO_60_REPORT.md`

## Final Writing Guidance

The safest paper stance is:

1. DAG-IG is a credit-assignment method for multimodal search agents.
2. The current implementation optimizes stage-1 visual/query behavior.
3. The offline Pix2Fact evidence-note result shows modest two-seed gains over Format-SFT.
4. KL-fixed training audits show the reward is not collapsed.
5. Remaining work is retrieval coverage and reader extraction.

Do not let the paper drift into claims about live web search, solved answer extraction, or final DPO/RL beyond the current GRPO evidence.
