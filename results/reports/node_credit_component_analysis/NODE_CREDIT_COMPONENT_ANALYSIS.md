# Node Credit Component Analysis

## 1. Scope

This analyzes the rollout logs from the paper-main GRPO runs. It is not a new training or evaluation run. It checks whether the DAG-IG node-level credits over visual, query, evidence, and answer nodes have non-trivial variation and align with retrieval/answer outcomes.

## 2. Reward Formula

For two-stage `paper_main_v1`, the training reward is:

```text
0.10 * format_credit
+ 0.15 * visual_credit
+ 0.40 * query_credit
+ 0.25 * evidence_credit
+ 0.35 * answer_credit
- leakage_penalty
- path_penalty
```

The stage1-only policy loss means the reward from the downstream reader/evidence path is assigned back to the visual/query stage.

## 3. Run-Level Reward Health

| Run | Rollouts | Groups | Reward mean | Reward std | Retrieval hit | Answer correct | Strict | AUC hit | AUC answer | AUC strict | Constant groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed42_main | 960 | 240 | 0.416 | 0.424 | 45.4% | 34.2% | 33.0% | 1.000 | 0.968 | 0.974 | 2 (0.8%) |
| seed43_confirm | 960 | 240 | 0.397 | 0.426 | 41.9% | 30.7% | 29.5% | 1.000 | 0.974 | 0.984 | 2 (0.8%) |
| goldfixed_control | 960 | 240 | 0.449 | 0.420 | 51.1% | 34.3% | 33.1% | 1.000 | 0.951 | 0.960 | 2 (0.8%) |

## 4. Component Statistics

| Run | Component | Mean | Std | Nonzero | Weighted contribution mean | AUC hit | AUC answer | AUC strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seed42_main | format | 0.099 | 0.006 | 100.0% | 0.010 | 0.493 | 0.500 | 0.499 |
| seed42_main | visual | 0.337 | 0.227 | 90.6% | 0.051 | 0.585 | 0.568 | 0.569 |
| seed42_main | query | 0.373 | 0.444 | 49.8% | 0.149 | 1.000 | 0.885 | 0.905 |
| seed42_main | evidence | 0.367 | 0.448 | 45.4% | 0.092 | 1.000 | 0.886 | 0.905 |
| seed42_main | answer | 0.334 | 0.469 | 34.2% | 0.117 | 0.861 | 1.000 | 1.000 |
| seed42_main | leakage_penalty | -0.001 | 0.014 | 0.3% | -0.001 | 0.497 | 0.495 | 0.495 |
| seed42_main | path_penalty | -0.001 | 0.011 | 1.7% | -0.001 | 0.501 | 0.496 | 0.498 |
| seed43_confirm | format | 0.099 | 0.005 | 100.0% | 0.010 | 0.494 | 0.501 | 0.502 |
| seed43_confirm | visual | 0.354 | 0.218 | 93.1% | 0.053 | 0.582 | 0.565 | 0.570 |
| seed43_confirm | query | 0.358 | 0.447 | 45.3% | 0.143 | 1.000 | 0.896 | 0.917 |
| seed43_confirm | evidence | 0.353 | 0.450 | 41.9% | 0.088 | 1.000 | 0.895 | 0.917 |
| seed43_confirm | answer | 0.299 | 0.455 | 30.7% | 0.105 | 0.849 | 1.000 | 1.000 |
| seed43_confirm | leakage_penalty | -0.001 | 0.018 | 0.5% | -0.001 | 0.502 | 0.499 | 0.501 |
| seed43_confirm | path_penalty | -0.001 | 0.010 | 1.6% | -0.001 | 0.507 | 0.506 | 0.506 |
| goldfixed_control | format | 0.099 | 0.005 | 100.0% | 0.010 | 0.488 | 0.486 | 0.487 |
| goldfixed_control | visual | 0.344 | 0.225 | 91.9% | 0.052 | 0.577 | 0.526 | 0.535 |
| goldfixed_control | query | 0.420 | 0.452 | 55.2% | 0.168 | 1.000 | 0.848 | 0.866 |
| goldfixed_control | evidence | 0.415 | 0.456 | 51.1% | 0.104 | 1.000 | 0.847 | 0.866 |
| goldfixed_control | answer | 0.335 | 0.469 | 34.3% | 0.117 | 0.820 | 1.000 | 1.000 |
| goldfixed_control | leakage_penalty | -0.001 | 0.014 | 0.3% | -0.001 | 0.497 | 0.495 | 0.495 |
| goldfixed_control | path_penalty | -0.001 | 0.012 | 1.4% | -0.001 | 0.506 | 0.503 | 0.503 |

## 5. Group Top-Bottom Credit Signal

| Run | Top hit | Bottom hit | Top answer | Bottom answer | Top strict | Bottom strict | Reward delta | Query delta | Evidence delta | Answer delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed42_main | 63.3% | 27.1% | 52.5% | 15.4% | 50.4% | 15.4% | 0.389 | 0.349 | 0.347 | 0.357 |
| seed43_confirm | 56.2% | 24.2% | 45.8% | 12.5% | 43.8% | 12.1% | 0.341 | 0.288 | 0.291 | 0.323 |
| goldfixed_control | 69.2% | 29.6% | 53.3% | 13.8% | 51.7% | 13.3% | 0.404 | 0.363 | 0.366 | 0.388 |

## 6. Interpretation

The node credits are not collapsed. Query and evidence credits are highly aligned with retrieval-hit labels, while answer credit is most aligned with strict success. Group top-bottom comparisons show that the highest-reward samples in each GRPO group have much higher retrieval and strict rates than the lowest-reward samples. This supports using DAG-IG as node-level credit assignment for the stage1 policy, even though it should not be interpreted as causal counterfactual intervention evidence and the final system is still limited by retrieval misses and reader errors.

## 7. Paper Use

This report is suitable as an internal source for a paper ablation/diagnostic paragraph: reward components are defined at visual/query/evidence/answer nodes, have measurable variance, and predict downstream support/strict success. It should be paired with the consolidated result table rather than used as a standalone performance claim.
