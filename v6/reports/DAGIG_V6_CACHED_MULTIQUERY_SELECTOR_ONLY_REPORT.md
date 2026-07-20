# Cached Multi-Query Evidence v2 Selector-Only Audit

## Scope

This is the one-shot internal selector-only evaluation frozen before private labels were opened. It uses cached real-search states only, performs no API calls, trains no generator, and does not open dev/test.

## Methods

| Method | Expected terminal | Support | Expected strict | Mode strict | Top tie | Strategies |
|---|---:|---:|---:|---:|---:|---:|
| no_credit | 0.154225 | 24.79% | 13.48% | 14.71% | 100.00% | 1 |
| local_ig | 0.197051 | 28.99% | 15.84% | 17.23% | 0.00% | 5 |
| outcome | 0.194084 | 29.41% | 15.48% | 16.39% | 0.00% | 5 |
| dagig | 0.199783 | 28.57% | 16.10% | 17.23% | 0.00% | 5 |

## Paired DAG-IG Comparisons

### DAG-IG vs no_credit

- top-action disagreement: 71.85%
- expected_terminal_value: delta=0.045558; gain/loss/tie=171/0/67; sample-clustered 95% CI=[0.035356, 0.057185]
- support: delta=0.037815; gain/loss/tie=16/7/215; sample-clustered 95% CI=[-0.021008, 0.095833]
- expected_strict: delta=0.026178; gain/loss/tie=20/14/204; sample-clustered 95% CI=[-0.003248, 0.062946]
- mode_strict: delta=0.025210; gain/loss/tie=9/3/226; sample-clustered 95% CI=[-0.004202, 0.059322]

### DAG-IG vs outcome

- top-action disagreement: 26.47%
- expected_terminal_value: delta=0.005698; gain/loss/tie=63/0/175; sample-clustered 95% CI=[0.003312, 0.008614]
- support: delta=-0.008403; gain/loss/tie=3/5/230; sample-clustered 95% CI=[-0.033333, 0.012605]
- expected_strict: delta=0.006189; gain/loss/tie=10/9/219; sample-clustered 95% CI=[-0.000374, 0.016704]
- mode_strict: delta=0.008403; gain/loss/tie=2/0/236; sample-clustered 95% CI=[0.000000, 0.021008]

### DAG-IG vs local_ig

- top-action disagreement: 16.39%
- expected_terminal_value: delta=0.002731; gain/loss/tie=39/0/199; sample-clustered 95% CI=[0.001493, 0.004137]
- support: delta=-0.004202; gain/loss/tie=2/3/233; sample-clustered 95% CI=[-0.021368, 0.012712]
- expected_strict: delta=0.002660; gain/loss/tie=7/4/227; sample-clustered 95% CI=[-0.009660, 0.015378]
- mode_strict: delta=0.000000; gain/loss/tie=2/2/234; sample-clustered 95% CI=[-0.016807, 0.016807]

## Action Diversity

- no_credit: `{"serper_rank_top3": 238}`
- local_ig: `{"bge_top3": 59, "entity_condition_mismatch_top3": 40, "observable_low_support_top3": 32, "serper_rank_top3": 65, "support_diverse_top3": 42}`
- outcome: `{"bge_top3": 63, "entity_condition_mismatch_top3": 46, "observable_low_support_top3": 27, "serper_rank_top3": 61, "support_diverse_top3": 41}`
- dagig: `{"bge_top3": 55, "entity_condition_mismatch_top3": 42, "observable_low_support_top3": 32, "serper_rank_top3": 67, "support_diverse_top3": 42}`

## Gates

- complete_238_internal_states: `True`
- complete_40_internal_samples: `True`
- direct_posterior_argmax_only: `True`
- dagig_terminal_gain_vs_no_credit: `True`
- dagig_terminal_noninferior_outcome: `True`
- dagig_support_not_below_no_credit: `True`
- dagig_support_noninferior_outcome: `True`
- dagig_expected_strict_noninferior_no_credit: `True`
- dagig_expected_strict_noninferior_outcome: `True`
- dagig_mode_strict_noninferior_no_credit: `True`
- dagig_mode_strict_noninferior_outcome: `True`
- dagig_differs_from_outcome: `True`
- dagig_action_diversity: `True`
- public_target_leakage_audit_passed: `True`
- internal_not_used_for_fit_or_tuning: `True`
- new_search_calls_zero: `True`
- generator_training_not_run: `True`
- dev_sealed: `True`
- test_sealed: `True`

## Decision

`DAGIG_V6_CACHED_MULTIQUERY_SELECTOR_ONLY_GO`

Proceed to a matched scalar evidence scorer/ranker using policy-train only. Primary objective: listwise KL; weighted pairwise cardinal ranking is an ablation. Do not train a categorical generator.
