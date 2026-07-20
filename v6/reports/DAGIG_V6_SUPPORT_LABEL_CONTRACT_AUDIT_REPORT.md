# DAG-IG v6 Support Label Contract Audit

Decision: `DAGIG_V6_SUPPORT_LABEL_CONTRACT_INVALID`

## Scope

Policy-train only. Internal holdout, dev, and test were not used.

## Metrics

- policy_train_evidence_actions: `11795`
- selected_query_actions_with_semantic_scores: `2359`
- legacy_rule_exactly_reproduced: `True`
- legacy_support_reason_counts: `{'answer_phrase_only': 1473, 'both': 991, 'negative': 8668, 'positive_url': 663}`
- answer_type_by_reason: `{'address::answer_phrase_only': 25, 'address::both': 63, 'address::negative': 716, 'address::positive_url': 21, 'email::answer_phrase_only': 70, 'email::both': 74, 'email::negative': 830, 'email::positive_url': 1, 'phone_or_identifier::answer_phrase_only': 283, 'phone_or_identifier::both': 379, 'phone_or_identifier::negative': 2003, 'phone_or_identifier::positive_url': 95, 'short_numeric::answer_phrase_only': 380, 'short_numeric::both': 125, 'short_numeric::negative': 1334, 'short_numeric::positive_url': 181, 'text_or_entity::answer_phrase_only': 689, 'text_or_entity::both': 350, 'text_or_entity::negative': 3608, 'text_or_entity::positive_url': 343, 'time::answer_phrase_only': 26, 'time::negative': 177, 'time::positive_url': 22}`
- short_numeric_answer_phrase_only_positives: `380`
- high_semantic_legacy_negatives: `22`
- low_semantic_legacy_positives: `55`

## Finding

The legacy support label is `positive URL match OR normalized answer phrase occurs anywhere in title/snippet`.
The answer-phrase branch checks neither entity identity nor question conditions. Short numeric answers can therefore match incidental numbers in unrelated pages, while semantically equivalent addresses can be marked negative due to formatting or translation differences.
This label is suitable as a loose retrieval-hit proxy, but not as semantic evidence support for calibrating `P_support` or for a paper-facing support metric.

## Consequence

Previous evidence/query support gates must be treated as provisional and recomputed after a frozen semantic-support label contract is established. Runtime policies remain gold-free; gold-aware judging is allowed only for private supervision/evaluation labels.

Conflict examples: `/root/dagig_scratch/v6_full_dag/support_label_contract_audit_v1_fixed/v6_support_label_conflicts_policy_train_private.jsonl`
