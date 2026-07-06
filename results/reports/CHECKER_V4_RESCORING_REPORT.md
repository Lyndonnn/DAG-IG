# Checker v4 Rescoring Report

## Scope

Existing two-stage predictions were rescored with answer checker v4. When a v3 parser-repaired prediction existed, v4 used that file as input; otherwise it used the original two-stage prediction. No model generation, retrieval, training data, or evidence pool was changed.

## Main Runs

| stem | n | R@5 | answer correct | strict | strict delta | changed rows |
|---|---:|---:|---:|---:|---:|---:|
| `format_sft_two_stage_own_full_dev` | 98 | 52.0% | 42.9% | 40.8% | -2 | 3 |
| `format_sft_two_stage_own_full_test` | 64 | 46.9% | 34.4% | 34.4% | 0 | 0 |
| `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev` | 98 | 57.1% | 49.0% | 46.9% | -2 | 2 |
| `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test` | 64 | 51.6% | 40.6% | 40.6% | 0 | 0 |
| `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev` | 98 | 58.2% | 49.0% | 46.9% | -2 | 2 |
| `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test` | 64 | 50.0% | 39.1% | 39.1% | 0 | 0 |
| `paper_main_v1_goldfixed_scale60_s320_ckpt60_dev` | 98 | 57.1% | 50.0% | 48.0% | -2 | 2 |
| `paper_main_v1_goldfixed_scale60_s320_ckpt60_test` | 64 | 50.0% | 39.1% | 39.1% | 0 | 0 |

## Changed Rows

- files rescored: `47`
- files with any checker change: `32`
- changed rows total: `67`
- machine-readable changes: `outputs/dagig_paper_main_v1/reports/parser_checker_v4_rescore_changes.json`
- machine-readable summary: `outputs/dagig_paper_main_v1/reports/parser_checker_v4_rescore_summary.json`
