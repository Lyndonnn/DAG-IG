# Train Gold Corpus Coverage Report

## Scope

This audits whether each train sample has at least one `is_gold=true` support document in the BM25 train corpus. This matters because query/evidence credit and GRPO rewards cannot learn retrieval for samples whose support doc is absent or not marked gold.

## Summary

- train samples: `458`
- corpus docs: `610`
- samples with any corpus doc: `458`
- samples with gold doc: `417`
- samples missing gold doc: `41`

## Classification Counts

| class | count |
|---|---:|
| has_gold_doc | 417 |
| sample_docs_present_but_none_marked_gold | 41 |

## Decision

This is a data/corpus construction issue for the affected samples. Before more GRPO, rebuild or patch the train BM25 corpus so every train row with evidence URLs has at least one corresponding `is_gold=true` support doc, then rerun reward/coverage audits. Do not treat missing-gold samples as model failures.

## Artifacts

- cases: `outputs/dagig_paper_main_v1/reports/train_gold_corpus_coverage/train_missing_gold_doc_cases.jsonl`
- summary: `outputs/dagig_paper_main_v1/reports/train_gold_corpus_coverage/train_gold_corpus_coverage_summary.json`
