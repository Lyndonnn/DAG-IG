# DAG-IG Paper Main Evidence Brief

## Paper Position

The main method should be positioned as node-level DAG-IG credit for a two-stage multimodal search agent, optimized with grouped GRPO over the stage-1 policy that emits `visual_observation` and `search_query`. The fixed reader consumes the image, question, and top-5 BM25 evidence to produce `final_answer`.

DAG-SFT is not the main claim. Preference-style planner tuning and query reranking were useful diagnostics, but the paper-facing result is the DAG-IG GRPO agent.

## Main Claim

In the frozen Pix2Fact offline BM25 setting, DAG-IG node-level GRPO improves the Format-SFT two-stage agent on both dev and test, with a second seed confirming the recipe.

| method | dev R@5 | dev strict | test R@5 | test strict |
|---|---:|---:|---:|---:|
| Format-SFT | 52.0% | 42.9% | 46.9% | 34.4% |
| DAG-IG seed42 main | 57.1% | 49.0% | 51.6% | 40.6% |
| DAG-IG seed43 confirm | 58.2% | 49.0% | 50.0% | 39.1% |
| Goldfixed control | 57.1% | 50.0% | 50.0% | 39.1% |

## Effect Size

Seed42 improves strict success by `6.1%` on dev and `6.2%` on test. Against Format-SFT, seed42 has `8` dev strict wins versus `2` dev losses, and `5` test strict wins versus `1` test losses.

## Why The Reward Is Trustworthy

| run | reward AUC hit | reward AUC strict | constant groups | top strict | bottom strict |
|---|---:|---:|---:|---:|---:|
| seed42 | 1.000 | 0.974 | 2/240 | 50.4% | 15.4% |
| seed43 | 1.000 | 0.984 | 2/240 | 43.8% | 12.1% |
| goldfixed | 1.000 | 0.960 | 2/240 | 51.7% | 13.3% |

The query and evidence components have AUC(hit)=1.000 in the main seed42 reward analysis, and the answer component has AUC(strict)=1.000. The format term is intentionally low-variance and does not drive the ranking.

## Reproducibility Anchors

- main checkpoint: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60`
- seed confirmation: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/checkpoint-60`
- goldfixed control: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/checkpoint-60`
- main result table: `outputs/dagig_paper_main_v1/paper_assets/main_results_table.tex`
- node-credit table: `outputs/dagig_paper_main_v1/paper_assets/node_credit_diagnostic_table.tex`
- case studies: `outputs/dagig_paper_main_v1/paper_assets/case_studies/CASE_STUDY_SUMMARY.md`

## Claim Boundaries

- Do not claim DAG-SFT is the main method; it is a diagnostic/pretraining baseline.
- Do not claim the goldfixed control is the best model; it is a robustness/control run.
- Do not claim real web search generalization; all results use the frozen offline BM25 corpus.
- Do not claim answer extraction is solved. The remaining bottlenecks are retrieval misses and retrieval-hit-answer-wrong cases.
- Do not launch more same-recipe GRPO runs without a new mechanism. The current next step is paper writing plus targeted qualitative/error presentation.

## Paper-Completion Next Step

Write the paper around the current evidence chain: problem formulation, DAG-IG node credit, two-stage GRPO training, reward audit, main results, seed confirmation, goldfixed control, and failure analysis. Additional experiments should be limited to paper-essential presentation gaps, not new method branches.
