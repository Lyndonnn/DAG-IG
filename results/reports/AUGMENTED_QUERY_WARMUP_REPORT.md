# Augmented Query Warmup Report
## Scope
This tests whether train-only no-hit recovery queries improve stage-1 query-node warmup. It does not use dev/test labels for training and does not train a reader.

## Data
- base hit-vs-miss rows: `311`
- no-hit recovery rows: `34`
- augmented rows: `345`
- no-hit train samples recovered by mining: `34` / `124` (`27.4%`)
- recovered support-rank counts: `{'1': 28, '2': 5, '5': 1}`

## Evaluation
| Method | n | R@1 | R@3 | R@5 | answer | strict | format | retrieval miss | hit-answer-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Format-SFT full dev | 98 | 35.7% | 49.0% | 52.0% | 45.9% | 42.9% | 100.0% | 47 | 9 |
| Query-node SFT smoke + fixed reader | 98 | 37.8% | 50.0% | 55.1% | 48.0% | 44.9% | 100.0% | 44 | 10 |
| Augmented query-node SFT smoke + fixed reader | 98 | 43.9% | 53.1% | 56.1% | 49.0% | 48.0% | 100.0% | 43 | 8 |
| DAG-IG GRPO seed42 ckpt60 dev | 98 | 38.8% | 51.0% | 57.1% | 51.0% | 49.0% | 99.0% | 42 | 8 |
| DAG-IG GRPO seed43 ckpt60 dev | 98 | 40.8% | 53.1% | 58.2% | 51.0% | 49.0% | 99.0% | 41 | 9 |
| Query-warm GRPO ckpt30 + fixed Format reader | 98 | 40.8% | 55.1% | 59.2% | 51.0% | 48.0% | 100.0% | 40 | 11 |

## Interpretation
- Compared with the earlier query-node warmup, augmented warmup improves strict from `44.9%` to `48.0%` and keeps hit-answer-wrong at `8`.
- It still trails the current seed42 main checkpoint on dev strict (`48.0%` vs `49.0%`) and R@5 (`56.1%` vs `57.1%`).
- The data change is useful but insufficient as a standalone adapter. Its value is as a cleaner initialization for exactly one controlled GRPO run, not as another branch to tune indefinitely.

## Decision
Run one short paper-main v1 GRPO initialized from `query_node_sft_aug_format_init_smoke20`, using the same two-stage stage1-loss settings as the current main recipe. Evaluate dev only first. Promote or run test only if dev strict exceeds the current `49.0%` or if retrieval improves without increasing hit-answer-wrong.

## Artifacts
- no-hit mining: `outputs/dagig_paper_main_v1/reports/nohit_query_candidate_mining/NOHIT_QUERY_CANDIDATE_MINING_REPORT.md`
- augmented data: `outputs/dagig_paper_main_v1/query_node_sft_aug/query_node_sft_aug_train.jsonl`
- augmented adapter: `outputs/dagig_paper_main_v1/checkpoints/query_node_sft_aug_format_init_smoke20`
- machine summary: `outputs/dagig_paper_main_v1/reports/augmented_query_warmup_summary.json`
