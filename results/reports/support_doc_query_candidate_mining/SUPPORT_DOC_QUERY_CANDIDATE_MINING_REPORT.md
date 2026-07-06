# Support-Document Query Candidate Mining Report

## Scope

This is train-only query mining for the no-hit samples that simple clean query recipes did not recover. It uses train support-document title/url/domain/text fields to build lexical query candidates, filters answer leakage, and evaluates only against the train BM25 corpus. It does not use dev/test labels or oracle trajectories.

## Results

- original no-hit train samples: `124`
- recovered by previous clean recipes: `34`
- remaining target samples: `90`
- generated candidate rows: `426`
- answer-leak candidates filtered: `22`
- newly recovered samples: `48` (`53.3%` of remaining)
- cumulative recovered no-hit samples: `82` / `124` (`66.1%`)
- new support-rank counts: `{'1': 48}`

## Candidate Source Counts

| source | candidates | hit candidates |
|---|---:|---:|
| support_doc_terms+semantic_anchor | 42 | 42 |
| support_doc_terms+intent | 36 | 36 |
| support_doc_terms+semantic_anchor+intent | 33 | 33 |
| support_url+semantic_anchor | 33 | 32 |
| support_title+semantic_anchor | 37 | 30 |
| support_doc_terms+hf_search_query | 27 | 27 |
| support_url+intent | 27 | 25 |
| support_doc_terms+hf_search_query+intent | 24 | 24 |
| support_title+intent | 30 | 24 |
| support_domain+semantic_anchor | 30 | 23 |
| support_url+hf_search_query | 23 | 22 |
| support_title+hf_search_query | 24 | 19 |
| support_domain+hf_search_query | 21 | 15 |
| support_doc_terms+visible_text | 10 | 10 |
| support_doc_terms+visible_text+intent | 8 | 8 |
| support_title+visible_text | 8 | 7 |
| support_domain+visible_text | 7 | 6 |
| support_url+visible_text | 6 | 5 |

## Decision

A large part of the remaining no-hit gap is recoverable when train support-document lexical fields are allowed. This suggests the core bottleneck is candidate query generation rather than an unrecoverable BM25/corpus mismatch. Use this as supervised train-only evidence to design a cleaner query-candidate generator, but keep it labeled separately because it is support-doc-derived.

## Artifacts

- candidates: `outputs/dagig_paper_main_v1/reports/support_doc_query_candidate_mining/support_doc_query_candidates.jsonl`
- best queries: `outputs/dagig_paper_main_v1/reports/support_doc_query_candidate_mining/support_doc_best_queries.jsonl`
- summary: `outputs/dagig_paper_main_v1/reports/support_doc_query_candidate_mining/support_doc_query_mining_summary.json`
