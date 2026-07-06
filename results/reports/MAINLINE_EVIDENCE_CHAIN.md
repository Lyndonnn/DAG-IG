# Mainline Evidence Chain

This is the single-entry audit trail for the DAG-IG / Pix2Fact paper mainline. It links the frozen data/corpus, unified rollout schema, node-level credit audit, selected GRPO checkpoint, and final dev/test result.

- created_at_utc: `2026-07-05T23:13:17.894333+00:00`
- overall pass: `True`

## Stages

| stage | passed | claim | evidence |
|---|---:|---|---|
| data_and_corpora | `True` | GRPO train/dev/test data and frozen BM25 train/eval corpora are present and counted. | `outputs/dagig_grpo_main/derived_assets/derived_manifest.json`<br>`outputs/dagig_grpo_main/derived_assets` |
| rollout_schema | `True` | Unified rollouts expose visual, query, evidence, answer, metrics, and node-credit fields. | `outputs/dagig_paper_main_v1/protocol/PAPER_MAIN_V1_SCHEMA.md`<br>`outputs/dagig_paper_main_v1/rollouts/train_rollouts_unified_scored.jsonl` |
| reward_audit | `True` | DAG-IG node-level reward is discriminative and non-collapsed before main GRPO training. | `outputs/dagig_paper_main_v1/reports/node_credit_component_analysis/node_credit_component_summary.json`<br>`outputs/dagig_paper_main_v1/paper_assets/node_credit_diagnostic_table.tex` |
| main_grpo_training | `True` | The selected two-stage DAG-IG GRPO checkpoint trained successfully under the paper-main config. | `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/grpo_run_config.json`<br>`outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/grpo_train_summary.json`<br>`outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60` |
| main_dev_test_result | `True` | The selected DAG-IG checkpoint improves over Format-SFT on both dev and test strict success and R@5. | `outputs/dagig_paper_main_v1/reports/paper_main_v1_consolidated_results.json`<br>`outputs/dagig_paper_main_v1/paper_assets/main_results_table.tex` |

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

## Reward Audit

| run | hit AUC | strict AUC | constant groups | top strict | bottom strict | query AUC hit | evidence AUC hit | answer AUC strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed42_main | 1.000 | 0.974 | 0.8% | 50.4% | 15.4% | 1.000 | 1.000 | 1.000 |
| seed43_confirm | 1.000 | 0.984 | 0.8% | 43.8% | 12.1% | 1.000 | 1.000 | 1.000 |
| goldfixed_control | 1.000 | 0.960 | 0.8% | 51.7% | 13.3% | 1.000 | 1.000 | 1.000 |

## Selected Checkpoint

- checkpoint: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60`
- status: `success`
- optimizer_steps: `60`
- micro_steps: `240`
- constant_reward_groups: `2`
- two_stage_loss_scope: `stage1`
- kl_coef: `0.1`

## Main Result

| split | Format-SFT strict | DAG-IG seed42 strict | seed42 strict gain | Format-SFT R@5 | DAG-IG seed42 R@5 | seed42 R@5 gain | seed43 strict gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | 42.9% | 49.0% | 6.1 pts | 52.0% | 57.1% | 5.1 pts | 6.1 pts |
| test | 34.4% | 40.6% | 6.2 pts | 46.9% | 51.6% | 4.7 pts | 4.7 pts |

## Boundary

This chain supports the current paper mainline only: DAG-IG node-level GRPO for a two-stage multimodal search agent. It does not promote DAG-SFT trace imitation, query reranking, no-teacher fusion, or broad answer repair to the main method.

