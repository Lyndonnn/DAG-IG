# Cached Multi-Query Evidence v2 Scalar Ranker Audit

## Scope

Four matched state-action scalar rankers were trained on policy-train only. Internal predictions were produced only after all models were frozen. No generator, API call, dev, or test was used.

## Target Fidelity

| Method | Mean TV | Forward KL | Top agreement | High-margin agreement |
|---|---:|---:|---:|---:|
| no_credit | 0.003833 | 0.000053 | 21.43% | - |
| local_ig | 0.136911 | 0.078450 | 26.89% | 20.69% |
| outcome | 0.167331 | 0.089825 | 22.69% | 20.99% |
| dagig | 0.134293 | 0.076034 | 23.53% | 23.46% |

## Executed Ranker Selections

| Method | Expected terminal | Support | Expected strict | Mode strict | Strategies |
|---|---:|---:|---:|---:|---:|
| no_credit | 0.149085 | 25.63% | 12.06% | 13.45% | 5 |
| local_ig | 0.161493 | 30.25% | 14.66% | 16.81% | 4 |
| outcome | 0.160725 | 29.41% | 15.15% | 17.23% | 4 |
| dagig | 0.160342 | 29.83% | 14.72% | 16.81% | 4 |

## Paired DAG-IG Comparisons

### DAG-IG vs no_credit

- top-action disagreement: 73.95%
- expected_terminal_value: delta=0.011257; gain/loss/tie=101/75/62; 95% CI=[0.001720, 0.023160]
- support: delta=0.042017; gain/loss/tie=16/6/216; 95% CI=[-0.004167, 0.087500]
- expected_strict: delta=0.026560; gain/loss/tie=21/15/202; 95% CI=[-0.004031, 0.057856]
- mode_strict: delta=0.033613; gain/loss/tie=11/3/224; 95% CI=[0.000000, 0.067227]

### DAG-IG vs outcome

- top-action disagreement: 25.21%
- expected_terminal_value: delta=-0.000382; gain/loss/tie=28/32/178; 95% CI=[-0.004238, 0.003271]
- support: delta=0.004202; gain/loss/tie=3/2/233; 95% CI=[-0.012500, 0.025424]
- expected_strict: delta=-0.004344; gain/loss/tie=0/8/230; 95% CI=[-0.012831, -0.000030]
- mode_strict: delta=-0.004202; gain/loss/tie=0/1/237; 95% CI=[-0.012712, 0.000000]

### DAG-IG vs local_ig

- top-action disagreement: 2.94%
- expected_terminal_value: delta=-0.001150; gain/loss/tie=2/5/231; 95% CI=[-0.002526, -0.000034]
- support: delta=-0.004202; gain/loss/tie=1/2/235; 95% CI=[-0.017094, 0.008475]
- expected_strict: delta=0.000573; gain/loss/tie=1/0/237; 95% CI=[0.000000, 0.001732]
- mode_strict: delta=0.000000; gain/loss/tie=0/0/238; 95% CI=[0.000000, 0.000000]

## Gates

- all_four_rankers_complete: `True`
- same_238_internal_states: `True`
- dagig_target_tv: `True`
- dagig_top_action_agreement: `False`
- dagig_high_margin_top_action_agreement: `False`
- dagig_terminal_gain_vs_no_credit_ranker: `True`
- dagig_terminal_noninferior_outcome_ranker: `True`
- dagig_support_noninferior_no_credit_ranker: `True`
- dagig_support_noninferior_outcome_ranker: `True`
- dagig_expected_strict_noninferior_no_credit_ranker: `True`
- dagig_expected_strict_noninferior_outcome_ranker: `True`
- dagig_mode_strict_noninferior_no_credit_ranker: `True`
- dagig_mode_strict_noninferior_outcome_ranker: `True`
- dagig_action_diversity: `True`
- internal_used_once_after_all_models_frozen: `True`
- private_labels_used_only_by_final_auditor: `True`
- no_api_calls: `True`
- dev_sealed: `True`
- test_sealed: `True`

## Decision

`DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_NO_GO`

Freeze the evidence ranker and proceed to the next DAG node. Keep the direct posterior selector as the executable ceiling and the weighted pairwise objective as a later ablation, not a tuning fallback.
