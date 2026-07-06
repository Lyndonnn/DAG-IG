# 7B Reward Choice Decision

## Decision

Use the original `paper_main_v1` DAG-IG reward as the 7B mainline reward.

Do not use `reward_v3` as the main 7B reward.

## Rationale

The 7B mainline must preserve the same-backbone comparison against the 3B paper-main DAG-IG method. The original paper-main reward contains:

- format credit
- visual credit
- query credit
- evidence credit
- answer credit
- leakage penalty
- path/query-quality penalty

`reward_v3` adds verifier answer-support shaping on top of the old reward. That is useful as a diagnostic and future ablation, but it is not the same method as the 3B paper-main DAG-IG reward. Using it as the main 7B reward would confound the same-backbone comparison.

## Current Status

- `reward_v3` audit is archived as an optional/future ablation.
- `reward_v3` trainer support, if present, must not be used for the main 7B run.
- The trainer now requires explicit `--allow_reward_v3_ablation` when `--two_stage_reward_version v3` is requested.
- A previously started `qwen25vl7b_dagig_full_v3_smoke_10` run was stopped and is invalid for reporting.
- The valid 7B main reward remains `paper_main_v1`.

## Mainline Next Run

Run the 7B same-backbone mainline in this order:

1. Qwen2.5-VL-7B Format-SFT.
2. Qwen2.5-VL-7B DAG-IG full smoke using `paper_main_v1` reward.
3. Compare same-backbone results against external baselines only after the same-backbone smoke is settled.
4. Revisit `reward_v3` only if the original reward degenerates on 7B, and only as a separate ablation.

## Non-Mainline Artifacts

These artifacts are diagnostic only and must not be reported as the main 7B DAG-IG method:

- `outputs/dagig_7b_extension/reward_audit_v3/`
- `outputs/dagig_7b_extension/checkpoints/qwen25vl7b_dagig_full_v3_preflight_1/`
- `outputs/dagig_7b_extension/checkpoints/qwen25vl7b_dagig_full_v3_smoke_10/`

The partially executed `qwen25vl7b_dagig_full_v3_smoke_10` directory has no valid final summary or final adapter and should be treated as aborted.

## Safety Guard

The trainer rejects `--two_stage_reward_version v3` unless `--allow_reward_v3_ablation` is also passed. This keeps accidental 7B mainline GRPO runs on the original `paper_main_v1` reward path.
