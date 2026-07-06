# Critical Paper Audit 2026-07-06

This repository currently preserves the paper-main code/results, but the attached paper claim should be treated as **not final** until the following issues are addressed:

1. The training script's current KL term is not a true reference-model KL constraint; it acts like an advantage shift. Main GRPO should be rerun after a correct KL estimator is implemented.
2. Current reports justify checkpoint promotion using test strict/R@5. Main-result selection must be dev-only or reported as seed mean/range.
3. Answer checker v3 has confirmed AM/PM false positives and broad substring/numeric matches. A v4 checker and full rescore are required.
4. The draft says the reader is fixed, but main-table evaluation uses own-reader. Keep own-reader as main only if the paper explicitly says so and reports the fixed-reader control.
5. The offline BM25 corpus is a small annotated-evidence corpus, not generic noisy web retrieval. Corpus size and nature must be disclosed.
6. "Counterfactual" / "IG" language is too strong unless explicit node interventions are implemented. Otherwise, describe the current method as DAG-structured node-level heuristic credit.

This audit is mirrored in the full workspace at:

```text
outputs/dagig_paper_main_v1/reports/CRITICAL_PAPER_AUDIT_20260706.md
```

The code export has been adjusted so `scripts/dagig_grpo/02_train_grpo.py --help` no longer requires omitted 7B modules. The remaining issues are result-affecting and require rescore/retraining or paper-claim revision.
