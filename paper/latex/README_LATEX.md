# DAG-IG LaTeX Scaffold

This directory contains a minimal article-style paper scaffold for the current DAG-IG paper-main evidence chain.

## Files

- `main.tex`: standalone draft using the current paper-main results.
- `appendix.tex`: appendix with diagnostic branch summary and reproducibility notes.
- `diagnostic_branches_table.tex`: compact appendix table for discarded diagnostic branches.
- `algorithm_dagig_grpo.tex`: method algorithm box included by `main.tex`.
- `../main_results_table.tex`: main result table included by `main.tex`.
- `../node_credit_diagnostic_table.tex`: reward diagnostic table included by `main.tex`.
- `../figures/dagig_method_diagram.tex`: TikZ method diagram included by `main.tex`.
- `../figures/dagig_reward_equations.tex`: method equations included by `main.tex`.
- `references.bib`: BibTeX references used by `main.tex`.
- `Makefile`: compile/check helper.

## Compile

From this directory:

```bash
make check
make all
```

The current scaffold uses standard LaTeX packages: `booktabs`, `amsmath`, `graphicx`, `tikz`, and `hyperref`.

## Paper Position

This draft intentionally frames DAG-IG GRPO as the main method. DAG-SFT, DPO pilots, query reranking, multi-query fusion, and answer repair are diagnostic/appendix material, not the main claim.

The current main checkpoint remains:

```text
outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60
```
