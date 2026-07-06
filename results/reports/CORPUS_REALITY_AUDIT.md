# Corpus Reality Audit

## Scope

This audit describes the frozen BM25 corpora used by the paper-main experiments. These corpora are Pix2Fact-derived evidence notes with URLs/domains, not live web pages and not a live web-search environment.

## Corpus Summary

| corpus | docs | expected samples | samples with gold doc | gold-doc coverage | median tokens | mean tokens | gold docs with answer text |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_original | 610 | 458 | 417 | 91.0% | 6.0 | 9.1 | 77.2% |
| train_goldfixed | 610 | 458 | 458 | 100.0% | 6.0 | 9.1 | 70.3% |
| eval_devtest | 201 | 162 | 150 | 92.6% | 6.0 | 8.8 | 80.7% |

## Interpretation

- The evaluation corpus is a small frozen dev/test pool, not a broad web index.
- Evidence text is short annotation-style support text. The median whitespace token length is low, so the paper should not describe this as retrieval from noisy full web documents.
- Gold support notes often contain the answer string directly. Strict success should therefore be interpreted as a controlled offline evidence-acquisition + extraction metric, not a live-web QA score.
- The goldfixed train corpus fixes train-side gold labels only; dev/test corpora remain frozen.
- Dev/test strict success is bounded by the availability of gold/supporting documents in this frozen corpus:
  - dev: 92/98 samples with gold docs = 93.9% upper bound from gold-doc coverage.
  - test: 58/64 samples with gold docs = 90.6% upper bound from gold-doc coverage.

## Required Paper Wording

Use wording such as: `a frozen Pix2Fact evidence-note BM25 corpus with 201 dev/test documents`, not `live web search` or `noisy web documents`.
