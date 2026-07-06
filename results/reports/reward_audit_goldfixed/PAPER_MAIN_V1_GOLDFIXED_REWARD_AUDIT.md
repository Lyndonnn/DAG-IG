# Paper Main v1 Gold-Fixed Reward Audit

## Scope

This rescoring audit uses the fixed train BM25 corpus and recomputes paper_main_v1 node credits for existing train rollouts. It checks whether the reward remains discriminative before any future GRPO run.

## Summary

- rollouts: `14656`
- old reward mean/std: `0.2855` / `0.2963`
- goldfixed reward mean/std: `0.3117` / `0.3024`
- old/new hit rollouts: `6127` / `6801`
- old/new strict rollouts: `110` / `140`
- AUC(reward, hit): `0.999`
- AUC(reward, strict): `0.938`
- constant reward groups: `37` / `3664` (`1.0%`)

## Source Summary

| source | rollouts | hit rollouts | strict rollouts |
|---|---:|---:|---:|
| dagig_grpo_full | 3664 | 1721 | 39 |
| dagig_grpo_no_visual | 3664 | 1751 | 33 |
| outcome_grpo | 3664 | 1632 | 36 |
| trajectory_grpo | 3664 | 1697 | 32 |

## Decision

GO for reward health under the fixed train corpus: reward variance is non-trivial, hit/strict AUC are high, and constant-reward groups are low. This does not mean immediate GRPO is required; it means future GRPO should use the fixed corpus and updated mining files.

## Artifacts

- rescored rollouts: `outputs/dagig_paper_main_v1/reports/reward_audit_goldfixed/train_rollouts_paper_main_v1_goldfixed_rescored.jsonl`
- summary: `outputs/dagig_paper_main_v1/reports/reward_audit_goldfixed/paper_main_v1_goldfixed_reward_audit_summary.json`
