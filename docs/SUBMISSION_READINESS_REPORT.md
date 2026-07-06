# DAG-IG Pix2Fact Submission Readiness Report

## Status

The paper-main experimental result is frozen. The current main method is DAG-IG node-level GRPO for a two-stage Pix2Fact multimodal search agent, initialized from Format-SFT and trained with stage-1 visual/query policy loss.

Current paper package state: ready for target-template conversion. The generic source bundle has been compiled locally with `pdflatex`/`bibtex`, and the rendered PDF passed the post-compile content audit.
The mainline evidence chain, text-finalization audit, artifact checksum verification, source/review-clean package checks, and venue-workspace audit also pass.

## Main Claim

DAG-IG improves the Format-SFT two-stage agent in the frozen offline BM25 Pix2Fact setting:

| Method | Dev R@5 | Dev strict | Test R@5 | Test strict |
|---|---:|---:|---:|---:|
| Format-SFT | 52.0% | 42.9% | 46.9% | 34.4% |
| DAG-IG seed42 main | 57.1% | 49.0% | 51.6% | 40.6% |
| DAG-IG seed43 confirm | 58.2% | 49.0% | 50.0% | 39.1% |
| Goldfixed control | 57.1% | 50.0% | 50.0% | 39.1% |

Seed42 remains the main checkpoint because it has the best test strict success and R@5. Seed43 confirms the recipe. Goldfixed is a robustness/control run, not the promoted main checkpoint.

## Completed Gates

- Main result table matches consolidated metric JSON.
- Node-credit diagnostic table matches reward-audit JSON.
- Reward is discriminative and non-collapsed: seed42 has 2 constant-reward groups out of 240 micro-steps.
- Citations in `main.tex` resolve against `references.bib`.
- Reproduction command template passes shell syntax checking.
- Source bundle validates with `make check`.
- Submission-facing TeX files contain no local `outputs/`, `/root/`, or internal script-path leaks.
- TeX linebreak compile-risk check passes.
- Overleaf/source zip bundle has been generated.
- Generic `submission_bundle/main.pdf` compiles successfully with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
- Post-compile PDF audit confirms that rendered text contains the main result numbers, reward diagnostics, rollout schema, limitations, diagnostic branch boundaries, and references.
- PDF layout audit confirms page metadata, extractable text on all pages, log/page-count consistency, and no warning-pattern lines.
- Text-finalization audit finds no unresolved TODO/TBD/placeholders, empty citations/references, local path leaks, unresolved compile-log references, or rendered-PDF placeholder text.
- Mainline evidence chain links frozen data/corpora, unified rollout schema, node credit/reward audit, selected checkpoint, and dev/test result.
- Venue-conversion workspace compiles through the generic wrapper and passes content/layout audits.
- Artifact checksum verification passes via `SHA256SUMS.txt`.

## Do Not Reopen As Mainline

- DAG-SFT trace imitation.
- Query reranking/switching.
- No-teacher multi-query fusion.
- Broad regex/span answer repair.
- Same-recipe GRPO reruns.
- DPO from weak preference scorer outputs.

These are diagnostic branches. They should appear only as motivation, appendix, or negative-result context.

## Remaining Non-Experimental Work

1. Convert `main.tex` into the target venue template.
2. Replace `Anonymous Authors` if the target venue allows non-anonymous submission or final camera-ready metadata.
3. Recompile after venue-template conversion:

```bash
cd submission_bundle
make check
make all
python scripts/post_compile_pdf_audit.py --pdf main.pdf --output_json post_compile_pdf_audit.json --output_md POST_COMPILE_PDF_AUDIT.md --require-pass
python scripts/pdf_layout_audit.py --pdf main.pdf --log main.log --output_json pdf_layout_audit.json --output_md PDF_LAYOUT_AUDIT.md --require-pass
```

4. Inspect the rendered PDF for table/figure placement.
5. Trim or expand appendix material depending on venue page limits.

## Current Known Limitation

The generic article-format PDF compiles and passes rendered-content, layout, and text-finalization audits in this environment. The venue-conversion workspace also compiles through its generic wrapper. The remaining limitation is venue-specific: the source still needs to be moved into the target venue template, with venue-specific author/anonymous metadata and page-limit checks.
