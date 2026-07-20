# DAG-IG v6 Gold-Aware Support Label Independent Audit

## Decision

`DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_V2_INDEPENDENT_AUDIT_NO_GO`

This is a blinded model-based independent audit on 350 policy-train actions. The GPT runner saw the private reference answer and evidence needed for semantic adjudication, but did not see the local prediction, legacy label, audit category, dev, or test data.

## Overall

- n: `350`
- accuracy: `62.9%`
- balanced accuracy: `65.4%`
- precision: `79.7%`
- recall: `56.2%`
- specificity: `74.6%`
- F1: `66.0%`
- disagreements: `130`

## Answer-Type Strata

| group | n | accuracy | balanced accuracy | precision | recall |
|---|---:|---:|---:|---:|---:|
| address | 27 | 70.4% | 52.1% | 75.0% | 90.0% |
| email | 18 | 33.3% | 29.2% | 50.0% | 41.7% |
| phone_or_identifier | 65 | 56.9% | 50.0% | 80.0% | 61.5% |
| short_numeric | 94 | 76.6% | 70.1% | 76.2% | 48.5% |
| text_or_entity | 141 | 61.0% | 66.7% | 87.3% | 53.9% |
| time | 5 | 0.0% | n/a | n/a | 0.0% |

## Audit Strata

| group | n | accuracy | balanced accuracy | precision | recall |
|---|---:|---:|---:|---:|---:|
| legacy_both | 50 | 74.0% | 86.2% | 100.0% | 72.3% |
| legacy_url_only | 50 | 80.0% | 81.2% | 90.9% | 71.4% |
| other_phrase_only | 50 | 58.0% | 69.6% | 95.8% | 53.5% |
| semantic_reject_legacy_positive | 70 | 35.7% | 50.0% | n/a | 0.0% |
| semantic_repair_legacy_negative | 70 | 58.6% | 50.0% | 58.6% | 100.0% |
| short_numeric_phrase_only | 60 | 80.0% | 70.0% | 100.0% | 40.0% |

## Frozen Gates

- exact_independent_audit_sample_min: `PASS`
- balanced_accuracy: `FAIL`
- precision: `FAIL`
- recall: `FAIL`
- short_numeric_subset_accuracy: `FAIL`
- address_subset_accuracy: `FAIL`
- runner_blinded_to_local_labels: `PASS`
- dev_sealed: `PASS`
- test_sealed: `PASS`

## Use Contract

- downstream credit recomputation allowed: `False`
- runtime policy feature use allowed: `False`
- dev used: `False`
- test used: `False`
- Serper calls: `0`

The provisional labels are not permitted for downstream training or claims. Inspect disagreement strata, revise the support-teacher semantics, and evaluate a new untouched blinded sample without relaxing thresholds.
