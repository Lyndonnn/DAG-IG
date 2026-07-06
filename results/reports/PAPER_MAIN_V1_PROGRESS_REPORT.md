# DAG-IG Paper Main v1 Progress Report

## 1. Main Protocol

The paper-main protocol is now fixed as:

```text
image + question
-> visual_observation
-> search_query
-> retrieve top-k evidence
-> final_answer
```

The schema is written to `outputs/dagig_paper_main_v1/protocol/PAPER_MAIN_V1_SCHEMA.md`.

## 2. Reward Audit

Existing GRPO rollouts were rescored with node-level DAG-IG credit for visual, query, evidence, and answer nodes.

| item | value |
|---|---:|
| rescored rollouts | 14656 |
| reward groups | 3664 |
| constant reward groups | 40 / 3664 = 1.1% |
| query-credit AUC vs support hit | 1.000 |
| total-reward AUC vs strict success | 0.939 |
| decision | GO for small GRPO smoke |

This fixes the earlier constant-reward failure mode. The old step-13 pattern with about 78.8% constant groups should not be treated as representative after the paper-main v1 reward rewrite.

## 3. One-Stage Finding

A one-stage GRPO pilot was run from `format_sft` with `paper_main_v1` reward.

| split | R@5 | answer correct | strict |
|---|---:|---:|---:|
| dev | 46.9% | 4.1% | 3.1% |
| test | 42.2% | 1.6% | 1.6% |

Decision: stop using one-stage `visual/query/final_answer` generation as the main method. It asks the model to answer before seeing retrieved evidence, so it is structurally misaligned with the intended agent.

## 4. Two-Stage Implementation

`scripts/dagig_grpo/02_train_grpo.py` now supports `--two_stage_rollout`.

Two-stage training does:

```text
stage 1: image + question -> visual_observation + search_query
retrieve top-k BM25 evidence
stage 2: image + question + retrieved evidence -> final_answer
```

`scripts/dagig_grpo/05_eval_two_stage.py` was also fixed so the default evaluation uses the same checkpoint as its own reader. A fixed-reader run must now be requested explicitly.

## 5. Two-Stage Pilot

| model | dev20 R@5 | dev20 answer correct | dev20 strict | retrieval hit answer wrong |
|---|---:|---:|---:|---:|
| format_sft own-reader | 40.0% | 25.0% | 20.0% | 4 |
| two-stage GRPO smoke, 1 step | 45.0% | 30.0% | 25.0% | 4 |
| two-stage GRPO pilot, 10 steps | 35.0% | 30.0% | 25.0% | 2 |

Training health for the 10-step two-stage pilot:

| item | value |
|---|---:|
| optimizer steps | 10 |
| micro steps | 40 |
| constant reward groups | 2 / 40 = 5.0% |
| max GPU memory | 21.8 GB |

## 6. Current Diagnosis

The reward/trainer are now healthy enough for small pilots, and the two-stage protocol matches the paper method.

The current blocker is not format parsing and not reward constancy. It is query stability:

- two-stage GRPO improved answer extraction on retrieved-hit cases (`retrieval_hit_answer_wrong` dropped from 4 to 2 on dev20);
- but retrieval R@5 dropped from 40.0% to 35.0% on the same dev20 slice;
- therefore longer GRPO from `format_sft` is not justified yet.

## 7. Next Mainline Step

Do not return to one-stage GRPO.

Additional stage-1-only GRPO was tested after this report's first draft. The trainer now supports:

```bash
--two_stage_loss_scope both|stage1|reader
```

The stage-1-only setting computes full two-stage DAG-IG reward, but applies GRPO logprob/KL loss only to the `visual_observation/search_query` generation. This keeps the method aligned with node-level credit assignment and tries to avoid reader drift.

| model | dev20 R@5 | dev20 answer correct | dev20 strict | retrieval hit answer wrong | constant groups |
|---|---:|---:|---:|---:|---:|
| format_sft own-reader | 40.0% | 25.0% | 20.0% | 4 | - |
| two-stage GRPO smoke, 1 step | 45.0% | 30.0% | 25.0% | 4 | 0 / 4 |
| two-stage GRPO pilot, 10 steps, both loss | 35.0% | 30.0% | 25.0% | 2 | 2 / 40 |
| two-stage GRPO pilot, 10 steps, stage1 loss | 35.0% | 30.0% | 25.0% | 2 | 1 / 40 |

The stage-1-only loss improves training health slightly but does not recover query R@5 after 10 steps. The current evidence says the useful update is very short/early; longer GRPO quickly perturbs query retrieval.

Next, stabilize the stage-1 query policy before longer two-stage GRPO:

1. Use the strongest existing query initializer or query-SFT checkpoint as stage-1 init if it is adapter-compatible.
2. Keep the two-stage reader/evidence prompt.
3. Treat the 1-step two-stage smoke as an early-stop candidate and evaluate it on full dev/test.
4. Continue only if full dev/test shows retrieval does not regress and strict improves.
5. If the 1-step signal does not hold, add an explicit query-preservation/SFT regularizer before longer GRPO.

Paper-facing claim is not ready yet. The useful progress is that the method has been re-aligned with the real agent loop and the reward signal is no longer degenerate.

## 8. Full Dev/Test Early-Stop Check

The 1-step two-stage GRPO smoke was evaluated on full dev/test against the corrected `format_sft` own-reader baseline.

| model | split | R@5 | answer correct | strict | retrieval hit answer wrong |
|---|---|---:|---:|---:|---:|
| format_sft | dev | 52.0% | 41.8% | 39.8% | 12 |
| early-stop 1-step two-stage GRPO | dev | 53.1% | 42.9% | 40.8% | 12 |
| format_sft | test | 46.9% | 32.8% | 32.8% | 9 |
| early-stop 1-step two-stage GRPO | test | 45.3% | 32.8% | 32.8% | 8 |

Pairwise:

| split | strict gains | strict losses | strict net | R@5 gains | R@5 losses | R@5 net |
|---|---:|---:|---:|---:|---:|---:|
| dev | 1 | 0 | +1 | 1 | 0 | +1 |
| test | 1 | 1 | 0 | 0 | 1 | -1 |

Decision: the early-stop candidate is safe but not yet a paper win. It proves the corrected two-stage DAG-IG GRPO loop can run without collapse, but the effect is too small and test retrieval slightly drops.

Next mainline experiment: stronger query preservation, starting with stage1-only two-stage GRPO at higher KL. The criterion remains strict: do not scale unless retrieval is preserved and strict improves on full dev/test.

## 9. High-KL Query-Preserving Pilot

The high-KL stage1-only two-stage pilot uses:

```text
--two_stage_rollout
--two_stage_loss_scope stage1
--kl_coef 0.1
--max_steps 10
```

Training health:

| run | optimizer steps | micro steps | constant reward groups | max GPU GB |
|---|---:|---:|---:|---:|
| stage1loss_kl0.1_10step | 10 | 40 | 1 / 40 = 2.5% | 19.7 |

Evaluation:

| model | split | R@5 | answer correct | strict | retrieval-hit answer-wrong |
|---|---|---:|---:|---:|---:|
| format_sft | dev | 52.0% | 41.8% | 39.8% | 12 |
| early-stop 1-step | dev | 53.1% | 42.9% | 40.8% | 12 |
| stage1loss_kl0.1_10step | dev | 53.1% | 41.8% | 39.8% | 13 |
| format_sft | test | 46.9% | 32.8% | 32.8% | 9 |
| early-stop 1-step | test | 45.3% | 32.8% | 32.8% | 8 |
| stage1loss_kl0.1_10step | test | 48.4% | 35.9% | 35.9% | 8 |

Pairwise against corrected `format_sft`:

| split | strict gain | strict loss | strict net | R@5 gain | R@5 loss | R@5 net |
|---|---:|---:|---:|---:|---:|---:|
| dev | 2 | 2 | 0 | 4 | 3 | +1 |
| test | 2 | 0 | +2 | 1 | 0 | +1 |

Decision: this is the current best paper-main v1 candidate. It is not final-scale evidence yet, but it is the first corrected two-stage DAG-IG GRPO setting that preserves dev and improves test. The next run should keep the same recipe and add checkpointed early stopping, not change the reward again.
