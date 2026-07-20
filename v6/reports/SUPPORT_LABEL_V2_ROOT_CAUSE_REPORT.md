# DAG-IG v6 Support Label v2 Root-Cause Analysis

## Decision

`DAGIG_V6_SUPPORT_LABEL_V2_ROOT_CAUSE_CONFIRMED`

The v2 label teacher is not repairable by changing its 0.5 threshold. Against the frozen independent blinded reference, its score AUC is useful but its best diagnostic balanced accuracy remains far below the pre-registered 0.90 gate.

## Metrics

- frozen-threshold balanced accuracy: `0.6543`
- frozen-threshold precision/recall: `0.7975` / `0.5625`
- continuous-score AUC: `0.7944`
- diagnostic best balanced accuracy across all observed thresholds: `0.7376` at `0.119203` (analysis only; not reused)
- legacy-trigger balanced accuracy: `0.5236`
- disagreements: `130/350`

## Root Causes

- agreement: `220`
- local_false_negative_exact_or_normalized_answer_present: `87`
- local_false_negative_semantic_entailment_or_conversion: `11`
- local_false_positive_incidental_or_wrong_context_match: `2`
- local_false_positive_without_answer_support: `30`

False negatives include exact phone numbers, emails, addresses, and numeric facts explicitly stated in a selected snippet. False positives include missing answers, wrong numbers, wrong entities, and topical-but-non-entailing evidence. This rules out threshold calibration as the main repair.

## Required v3 Contract

1. Use a stronger structured semantic teacher, not next-token A/B logits.
2. Require every positive decision to identify a supporting document and return a verifiable evidence span or explicit derivation.
3. Run a small fresh teacher-versus-independent-auditor pilot before scoring the full 14,770-action universe.
4. Freeze a new untouched audit set; do not reuse these 350 items as the v3 quality gate.
5. Keep labels evaluation/value-supervision only; runtime policies must use a separately trained no-gold verifier.
