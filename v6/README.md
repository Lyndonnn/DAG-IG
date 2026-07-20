# DAG-IG v6 Development Snapshot

This directory contains the current complete-counterfactual-DAG development
snapshot. It is intentionally separate from the historical 3B GRPO release at
the repository root.

## Contents

- `dagig_causal/`: reusable DAG-IG, value, schema, retrieval, and policy code.
- `scripts/`: v6 scripts numbered 700-846, plus the current answer matcher and
  resumable full-v3 label runner.
- `specs/`: method specifications that led to the current exact posterior form.
- `reports/`: public summaries and audits; private labels and API logs are not
  included.

## Current Entry Point

The active label/value repair sequence is:

```text
823  audit legacy support contract
830-833  independently audit and diagnose the failed v2 teacher
834-836  freeze/run/audit the cited v3 teacher pilot
837-839  freeze/score/aggregate full v3 semantic labels
840-843  freeze/score/fit/audit the runtime no-gold support verifier
844-846  freeze/fit/audit corrected answer-action terminal values
```

Scripts 783-801 implement the cached multi-query evidence, query, visual, and
selector-only path. Their old support/strict outputs must not be reused after
the label-contract audit. They are retained to make the experimental history
and the upcoming corrected recomputation auditable.

Scripts 700-782 capture the immediate v6 value, policy, and distillation path
that motivated selector-only evaluation. They include NO-GO experiments and are
not all current recommended entry points.

## Reproduction Boundary

The scripts expect frozen manifests and action records generated in the full
experiment workspace. Those records may contain dataset-derived or private
evaluation fields and are not committed here. The repository release does not
contain API keys, private teacher labels, request logs, checkpoints, raw images,
or model caches.

See `../docs/CURRENT_RESEARCH_STATUS_2026-07-20.md` for the exact trust boundary
and next-step order.
