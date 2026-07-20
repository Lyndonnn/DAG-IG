# DAG-IG v6 Structured Support Teacher Pilot v3

## Decision

`DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_GO`

## Overall

- n: `400`
- accuracy: `97.5%`
- balanced accuracy: `97.2%`
- precision: `92.4%`
- recall: `96.6%`
- disagreements: `10`
- teacher/auditor citation validity: `100.0%` / `100.0%`

## Answer Types

| type | n | accuracy | balanced accuracy | precision | recall |
|---|---:|---:|---:|---:|---:|
| address | 44 | 97.7% | 98.5% | 91.7% | 100.0% |
| email | 45 | 97.8% | 93.8% | 100.0% | 87.5% |
| phone_or_identifier | 70 | 98.6% | 99.0% | 95.2% | 100.0% |
| short_numeric | 70 | 97.1% | 98.4% | 75.0% | 100.0% |
| text_or_entity | 159 | 96.9% | 96.2% | 92.7% | 95.0% |
| time | 12 | 100.0% | 100.0% | 100.0% | 100.0% |

## Gates

- exact_samples: `PASS`
- balanced_accuracy: `PASS`
- precision: `PASS`
- recall: `PASS`
- teacher_citation_validity: `PASS`
- auditor_citation_validity: `PASS`
- short_numeric_accuracy: `PASS`
- phone_or_identifier_accuracy: `PASS`
- email_accuracy: `PASS`
- address_accuracy: `PASS`
- auditor_blinded_to_teacher: `PASS`
- dev_sealed: `PASS`
- test_sealed: `PASS`

## Next Action

Freeze and score the full deduplicated 14,770-action support universe with the exact teacher contract, then run another untouched audit.
