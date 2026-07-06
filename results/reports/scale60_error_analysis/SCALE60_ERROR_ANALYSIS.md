# Scale60 Error Analysis

This analyzes the current main checkpoint `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60` without changing predictions or metrics.

## Summary

| split | n | strict | retrieval miss | hit-answer-wrong | answer correct without support | error file |
|---|---:|---:|---:|---:|---:|---|
| dev | 98 | 48 | 42 | 8 | 0 | `outputs/dagig_paper_main_v1/reports/scale60_error_analysis/dev_errors.jsonl` |
| test | 64 | 26 | 31 | 7 | 0 | `outputs/dagig_paper_main_v1/reports/scale60_error_analysis/test_errors.jsonl` |

## Retrieval Miss Subtypes

### dev

- `query_drift_from_teacher_search_intent`: `14`
- `missing_semantic_anchor`: `10`
- `retrieves_wrong_sample_cluster`: `7`
- `query_too_short_or_generic`: `6`
- `retrieval_miss_other`: `5`

### test

- `missing_semantic_anchor`: `13`
- `query_drift_from_teacher_search_intent`: `7`
- `retrieves_wrong_sample_cluster`: `5`
- `retrieval_miss_other`: `4`
- `query_too_short_or_generic`: `2`

## Hit-Answer-Wrong Subtypes

### dev

- `answer_extraction_wrong`: `5`
- `too_short_address`: `1`
- `wrong_type_numeric`: `1`
- `wrong_type_phone`: `1`

### test

- `answer_extraction_wrong`: `4`
- `wrong_type_numeric`: `2`
- `wrong_type_phone`: `1`

## Decision

The dominant remaining bottleneck is still retrieval miss, not answer extraction: ckpt60 has 42 dev and 31 test retrieval misses versus 8 dev and 7 test hit-answer-wrong cases. The next paper-facing work should therefore prioritize query/evidence node training and hard retrieval-miss analysis. Reader training should be revisited only after building a larger hard-context answer dataset.
