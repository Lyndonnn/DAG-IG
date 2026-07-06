# DAG-IG Pix2Fact Corrected Package Readiness Report

## Status

The corrected paper-main result package is internally consistent after the reviewer-audit fixes. The current main method is DAG-IG node-level GRPO for a two-stage Pix2Fact multimodal search agent, initialized from Format-SFT and trained with stage-1 visual/query policy loss.

Current paper package state: corrected core artifacts are ready for renewed paper editing and target-template conversion. The generic source bundle has been compiled locally in earlier checks, but the paper text should be recompiled after the corrected KL/checker/corpus wording changes.
The release verification script, core-fix validation, checksum artifacts, and current metric tables pass.

## Main Claim

DAG-IG improves the Format-SFT two-stage agent in the frozen offline Pix2Fact evidence-note BM25 setting. The clean headline is the KL-fixed two-seed mean, not the best test seed:

| Method | Dev R@5 | Dev strict | Test R@5 | Test strict |
|---|---:|---:|---:|---:|
| Format-SFT v4 | 52.0% | 40.8% | 46.9% | 34.4% |
| KL-fixed GRPO seed42 | 56.1% | 45.9% | 51.6% | 40.6% |
| KL-fixed GRPO seed43 | 56.1% | 45.9% | 48.4% | 37.5% |
| KL-fixed GRPO two-seed mean | 56.1% | 45.9% | 50.0% | 39.1% |

The corrected two-seed mean improves strict success over Format-SFT by +5.1 dev points and +4.7 test points. Fixed-reader control matches the own-reader two-seed strict result, so the gain is not explained by changing the reader. Paired tests are directionally positive but not conventionally significant, so the claim should remain modest.

## Completed Gates

- Main result table matches corrected KL-fixed metric JSON.
- Node-credit diagnostic table matches reward-audit JSON.
- KL-fixed training is non-collapsed: seed42 has 3 constant-reward groups out of 240 micro-steps and seed43 has 1 out of 240.
- The KL penalty uses the non-negative k3 estimator.
- Answer checker v4 blocks the audited AM/PM, substring-boundary, and numeric-range false positives.
- Fixed-reader control has been run for both KL-fixed seeds.
- Corpus reality audit documents the frozen Pix2Fact evidence-note corpus boundary.
- Citations in `main.tex` resolve against `references.bib`.
- Reproduction command template passes shell syntax checking.
- Source bundle validates with `make check`.
- Submission-facing TeX files contain no local `outputs/`, `/root/`, or internal script-path leaks.
- TeX linebreak compile-risk check passes.
- Overleaf/source zip bundle has been generated.
- Existing generic PDF/build audits are historical and should be rerun after final venue conversion.
- Mainline evidence chain has been corrected to link the KL-fixed two-seed result instead of the old-KL seed42 result.
- Venue-conversion workspace should be regenerated after the corrected paper text is finalized.
- Artifact checksum verification passes via `SHA256SUMS.txt`.

## Do Not Reopen As Mainline

- DAG-SFT trace imitation.
- Query reranking/switching.
- No-teacher multi-query fusion.
- Broad regex/span answer repair.
- Same-recipe GRPO reruns.
- DPO from weak preference scorer outputs.
- Old-KL seed42 as the headline result.
- Best-test-seed selection.
- Claims of causal counterfactual intervention without explicit intervention experiments.
- Claims of live web or noisy full-document retrieval generalization.

These are diagnostic branches. They should appear only as motivation, appendix, or negative-result context.

## Remaining Non-Experimental Work

1. Recompile the corrected generic paper source.
2. Convert `main.tex` into the target venue template.
3. Replace `Anonymous Authors` if the target venue allows non-anonymous submission or final camera-ready metadata.
4. Recompile after venue-template conversion:

```bash
cd submission_bundle
make check
make all
python scripts/post_compile_pdf_audit.py --pdf main.pdf --output_json post_compile_pdf_audit.json --output_md POST_COMPILE_PDF_AUDIT.md --require-pass
python scripts/pdf_layout_audit.py --pdf main.pdf --log main.log --output_json pdf_layout_audit.json --output_md PDF_LAYOUT_AUDIT.md --require-pass
```

5. Inspect the rendered PDF for table/figure placement.
6. Trim or expand appendix material depending on venue page limits.

## Current Known Limitation

The corrected package now passes code/result verification, but final submission readiness still depends on recompiling the corrected paper text and moving it into the target venue template. The current experimental limitation is substantive: the result is a small-sample offline evidence-note retrieval result with modest paired significance, not a live-web or large-scale SOTA claim.
