# Mainline Evidence Chain

This is the single-entry audit trail for the corrected DAG-IG / Pix2Fact paper mainline. It links the frozen data/corpus, unified rollout schema, node-level credit audit, KL-fixed GRPO reruns, fixed-reader control, and corrected dev/test result.

- created_at_utc: `2026-07-05T23:13:17.894333+00:00`
- overall pass: `True`

## Stages

| stage | passed | claim | evidence |
|---|---:|---|---|
| data_and_corpora | `True` | GRPO train/dev/test data and frozen BM25 train/eval corpora are present and counted. | `outputs/dagig_grpo_main/derived_assets/derived_manifest.json`<br>`outputs/dagig_grpo_main/derived_assets` |
| rollout_schema | `True` | Unified rollouts expose visual, query, evidence, answer, metrics, and node-credit fields. | `outputs/dagig_paper_main_v1/protocol/PAPER_MAIN_V1_SCHEMA.md`<br>`outputs/dagig_paper_main_v1/rollouts/train_rollouts_unified_scored.jsonl` |
| reward_audit | `True` | DAG-IG node-level reward is discriminative and non-collapsed before main GRPO training. | `outputs/dagig_paper_main_v1/reports/node_credit_component_analysis/node_credit_component_summary.json`<br>`outputs/dagig_paper_main_v1/paper_assets/node_credit_diagnostic_table.tex` |
| main_grpo_training | `True` | Two KL-fixed DAG-IG GRPO seeds trained successfully under the corrected paper-main config. | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed42/grpo_train_summary.json`<br>`outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed43/grpo_train_summary.json` |
| main_dev_test_result | `True` | The KL-fixed two-seed mean improves over Format-SFT on both dev and test strict success and R@5. | `results/reports/KLFIXED_GRPO_60_REPORT.md`<br>`results/metrics/klfixed_grpo_60_summary.json`<br>`results/tables/main_results_table.csv` |

## Data And Corpus Counts

| item | observed | expected |
|---|---:|---:|
| derived_grpo_train | 458 | 458 |
| derived_grpo_dev | 98 | 98 |
| derived_grpo_test | 64 | 64 |
| derived_bm25_train_docs | 610 | 610 |
| derived_bm25_eval_docs | 201 | 201 |

## Unified Rollout Schema

- rows: `14656`
- parsed_json_rate: `98.2%`
- source_runs: `{'outcome_grpo': 3664, 'trajectory_grpo': 3664, 'dagig_grpo_no_visual': 3664, 'dagig_grpo_full': 3664}`
- top_k_counts: `{'5': 14656}`
- invalid row examples: `[]`

## Corrected Training Health

| run | optimizer steps | micro steps | constant groups | constant rate |
|---|---:|---:|---:|---:|
| KL-fixed seed42 | 60 | 240 | 3 | 1.2% |
| KL-fixed seed43 | 60 | 240 | 1 | 0.4% |

The KL penalty uses the non-negative k3 estimator, and answer metrics use checker v4. Old-KL seed42/seed43 and goldfixed runs are diagnostics, not the corrected headline.

## Main Result

| split | Format-SFT strict | KL-fixed mean strict | strict gain | Format-SFT R@5 | KL-fixed mean R@5 | R@5 gain |
|---|---:|---:|---:|---:|---:|---:|
| dev | 40.8% | 45.9% | 5.1 pts | 52.0% | 56.1% | 4.1 pts |
| test | 34.4% | 39.1% | 4.7 pts | 46.9% | 50.0% | 3.1 pts |

Fixed-reader controls match the own-reader two-seed strict result: 45.9% dev and 39.1% test.

## Boundary

This chain supports the corrected paper mainline only: DAG-IG node-level GRPO for a two-stage multimodal search agent in a frozen Pix2Fact evidence-note retrieval setting. It does not promote DAG-SFT trace imitation, query reranking, no-teacher fusion, broad answer repair, old-KL results, or best-test-seed selection to the main method.
