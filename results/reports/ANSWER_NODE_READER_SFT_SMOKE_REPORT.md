# Answer Node Reader SFT Smoke Report

## Purpose

This step tested whether the current answer bottleneck can be fixed cheaply after the clean DAG-IG query-policy gain. The query generator is kept fixed at:

`outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_medium30/checkpoint-30`

Only the reader/verifier side was changed.

## Lightweight Verifier

Script:

`scripts/dagig_paper_main/03_learned_answer_verifier.py`

Train data came from ckpt30 train rollouts only. Actual inference features exclude gold answer, gold doc labels, support labels, and correctness.

| split | before strict | after strict | replacements | hit-answer-wrong before | after |
|---|---:|---:|---:|---:|---:|
| dev | 48.0% | 48.0% | 0 | 9 | 9 |
| test | 39.1% | 39.1% | 0 | 7 | 7 |

Decision: no gain. Do not add lightweight post-hoc verifier/repair to the main method.

## Reader SFT Data

Script:

`scripts/dagig_paper_main/04_build_reader_sft_data.py`

Aligned data output:

`outputs/dagig_paper_main_v1/reader_sft_aligned/reader_sft_train.jsonl`

Data summary:

- train rollouts scanned: `480`
- reader SFT rows: `183`
- unique samples: `70`
- top-k: `5`
- support rank counts: `{"1": 132, "2": 27, "3": 6, "4": 12, "5": 6}`
- no dev/test labels used
- no teacher/oracle query injected
- train gold answer used only as supervised reader target

## Reader SFT Smoke

Two 20-step reader smoke runs were tested from Format-SFT init:

| reader | prompt alignment | Dev R@5 | Dev answer | Dev strict | hit-answer-wrong |
|---|---|---:|---:|---:|---:|
| ckpt30 own reader | current baseline | 57.1% | 50.0% | 48.0% | 9 |
| reader_sft_format_init_smoke20 | query included in train prompt only | 57.1% | 44.9% | 41.8% | 15 |
| reader_sft_aligned_format_init_smoke20 | eval prompt aligned | 57.1% | 45.9% | 44.9% | 12 |

The aligned reader is better than the mismatched reader, but still below the ckpt30 own reader. Test was not run for the aligned reader because dev already failed the go criterion.

## Diagnosis

The current 183-row reader dataset is too narrow and biased toward easy support contexts. It teaches concise JSON extraction, but it does not outperform the existing reader on dev. This is a data/construction issue, not evidence that the answer node is irrelevant.

The clean paper result should therefore remain:

- Format-SFT: dev/test strict `42.9% / 34.4%`
- DAG-IG two-stage GRPO ckpt30: dev/test strict `48.0% / 39.1%`

The demonstrated gain is still from DAG-IG query/evidence policy, not reader improvement.

## Next Mainline Decision

Do not continue broad repair rules or tiny reader SFT. For the paper path:

1. Keep ckpt30 as the current main method checkpoint.
2. Scale or improve clean rollout/query/evidence training before more GRPO.
3. Rebuild reader training data only when it includes more diverse hard contexts: retrieval-hit-answer-wrong, noisy top-k, and answer-type-balanced examples.
4. The next GRPO run should preserve the successful recipe: two-stage rollout, stage1-only loss, KL=0.1, and `paper_main_v1` reward.
