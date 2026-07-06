# Gold-Fixed Train Corpus Manifest

## Scope

This builds a fixed train BM25 corpus without overwriting the original. For train samples with no `is_gold=true` doc, same-sample docs whose URL appears in the row's `evidence_urls` are marked as gold. The rule is uniform and train-only.

## Summary

- input corpus: `outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl`
- output corpus: `outputs/dagig_paper_main_v1/derived_assets/bm25_train_corpus_goldfixed.jsonl`
- train samples: `458`
- samples with gold before: `417`
- samples with gold after: `458`
- missing gold after: `0`
- docs fixed: `56`
- fix reason counts: `{'evidence_url_missing_gold_flag': 51, 'answer_match_evidence_url': 5}`

## Decision

Use this fixed corpus for future train-side reward/coverage audits and any future GRPO runs. Do not compare future train-side reward statistics to old runs without noting the corpus fix. Dev/test corpora are not modified by this script.

## Artifacts

- fixed corpus: `outputs/dagig_paper_main_v1/derived_assets/bm25_train_corpus_goldfixed.jsonl`
- changed docs: `outputs/dagig_paper_main_v1/reports/train_gold_corpus_coverage/goldfixed_doc_changes.jsonl`
- machine summary: `outputs/dagig_paper_main_v1/reports/train_gold_corpus_coverage/goldfixed_train_corpus_summary.json`
