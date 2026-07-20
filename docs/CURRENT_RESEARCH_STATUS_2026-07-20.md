# DAG-IG Current Research Status

Date: 2026-07-20

## Executive Summary

DAG-IG is a node-level credit-assignment method for long-horizon multimodal
search agents. The current executable trajectory is:

```text
image + question
  -> visual action
  -> structured query action
  -> cached real search
  -> evidence-set action
  -> frozen answer policy
  -> final answer
```

The project is not yet at a paper-final result. The old 3B KL-fixed GRPO result
remains a useful historical baseline, but it is not the final proof of DAG-IG.
The current paper-facing direction is a complete counterfactual DAG evaluated
first as an exact posterior selector. Policy distillation and GRPO/GDPO are
deferred until the selector itself passes all nodes under corrected labels.

The most important recent finding is negative but productive: the legacy
evidence-support label was invalid. It marked evidence positive when either a
known URL matched or the normalized answer string appeared anywhere in a
snippet. This created false positives for short numbers and false negatives for
semantic equivalents. All support/strict-success results derived from that
label have therefore been downgraded to provisional diagnostics.

A replacement structured, cited semantic-support teacher passed an independent
400-state pilot with 97.17% balanced accuracy, 92.39% precision, and 96.59%
recall. Full labeling is currently running. Official dev/test remain sealed.

## Method That Is Being Proven

Let `mu_n(a | s)` be the frozen behavior prior at node `n`, and let `Y` denote
terminal success. DAG-IG uses the exact control-as-inference posterior:

```text
q_n(a | s, Y=1) = mu_n(a | s) V_{n+1}(s,a) / V_n(s)
V_n(s)           = sum_a mu_n(a | s) V_{n+1}(s,a)
IG_n(s,a)        = log V_{n+1}(s,a) - log V_n(s)
```

The edge credits telescope along a complete path. Each method receives the
same legal actions, cached search results, evidence sets, answer policy, and
compute budget:

- **No-credit:** frozen behavior prior.
- **Local-IG-M:** local/immediate success proxy.
- **Outcome:** equal-budget sampled terminal outcome estimate.
- **DAG-IG:** exact backward expectation over the frozen descendant policies.

The direct posterior selector is the primary method-development instrument. It
isolates whether the credit formulation works without conflating it with a
generator's ability to learn a difficult distribution. A learned ranker or
Node-GDPO/GRPO policy is an optional later stage, not the current gate.

## Data and Executable State Space

The current v6 method-development universe contains:

| Asset | Count | Status |
|---|---:|---|
| Train-development samples | 198 | Frozen |
| Policy-train / internal samples | 158 / 40 | Frozen |
| Complete cached real-search query actions | 2,954 | Frozen |
| Executable evidence actions | 14,770 | Five legal actions per query state |
| Deduplicated semantic evidence states | 12,365 | Full v3 labeling universe |
| Shared answer-policy actions | 41,273 | Frozen |
| Official dev/test | Sealed | Not used for current tuning |

The five evidence interventions are:

1. `serper_rank_top3`
2. `bge_top3`
3. `support_diverse_top3`
4. `observable_low_support_top3`
5. `entity_condition_mismatch_top3`

Runtime action records may contain only observable state/action information.
Gold answers, qrels, target documents, support labels, final correctness, and
teacher/oracle fields are private supervision/evaluation data and are forbidden
from runtime selector features.

## Completed Experimental Phases

### 1. Grounded Pix2Fact trajectory construction

The early direct numeric-bbox route failed. Replacing it with a model-generated
grounding expression followed by official GroundingDINO substantially improved
localization. This established that tool-grounded expressions were more viable
than autonomous bbox regression, but end-to-end answer accuracy remained low.

### 2. Retrieval pipeline decomposition

The v2 pipeline separated planner query generation, frozen BM25 retrieval, and
evidence-conditioned answering. On the original verified seed package, teacher
and learned queries retrieved useful evidence while closed-book answering
remained weak. This identified retrieval and answer extraction as separable
bottlenecks.

### 3. 414-sample diagnostic expansion

The clean training split was expanded to 414 samples while preserving the
frozen 74/48 dev/test split. Counterfactual query/evidence gates were predictive,
but direct planner tuning, query reranking, switch selection, and strict
no-teacher evidence fusion did not consistently beat the stable DAG-SFT query
pipeline. A provenance-filtering bug in the clean fusion pool was found and
fixed uniformly; the corrected pool achieved 100% DAG-SFT top-5 coverage, yet
answer extraction remained the dominant bottleneck.

### 4. Small SFT/DPO and offline-evaluation pilots

Clean SFT/DPO packages were audited for missing images and oracle actions.
Outcome-SFT, verbose DAG-SFT, and DAG-lite SFT smoke runs established a fair
offline retrieval harness and fixed answer normalization. These experiments
showed that SFT target loss is not comparable across output formats and that
DAG-SFT is an initializer/ablation, not the main DAG-IG validation.

### 5. Historical 3B GRPO result and external audit

The corrected historical result used non-negative k3 KL, checker v4, two seeds,
and a fixed-reader control. Its two-seed mean improved strict success over
Format-SFT by 5.1 dev points and 4.7 test points. This remains a modest baseline,
not a statistically settled top-conference claim. The audit also exposed that
the old evaluation corpus was only a 201-document evidence-note corpus with a
median length of six whitespace tokens, which is too easy to serve as the final
external retrieval setting.

### 6. Full executable counterfactual DAG v6

The project then moved from reward-component naming to executable node
interventions. It built explicit visual, query, evidence, and answer action
universes; frozen behavior priors; cached real search; answer-policy backups;
exact posterior normalization; and telescoping audits. Dev/test were kept
sealed during this method development.

### 7. Evidence selector and failed distillation

Under the then-current labels, the cached multi-query direct DAG-IG posterior
selector passed its one-shot internal gate. The scalar ranker failed to preserve
the target posterior: top-action agreement was only 23.53%. This justified the
current strategy of proving the complete direct selector before attempting
distillation. These selector outcome numbers are now provisional because their
support/strict labels used the invalid legacy contract.

### 8. Support-label audit and repair

The legacy label contract was reproduced exactly and rejected. A local Qwen
next-token support teacher also failed a blinded 350-state independent audit:
65.43% balanced accuracy, 79.75% precision, and 56.25% recall. Threshold tuning
could not repair it; its best diagnostic balanced accuracy was only 73.76%.

The replacement v3 teacher requires every positive decision to identify
supporting documents and return a verifiable copied span or explicit derivation.
On a fresh, disjoint 400-state pilot:

| Metric | Result |
|---|---:|
| Accuracy | 97.50% |
| Balanced accuracy | 97.17% |
| Precision | 92.39% |
| Recall | 96.59% |
| Teacher citation validity | 100% |
| Independent auditor citation validity | 100% |

All preregistered numeric, phone/identifier, email, address, and citation gates
passed. The full 12,365-state labeling run is resumable and in progress.

## Trust Boundary

| Result or artifact | Current use |
|---|---|
| Frozen action universes, cached search, behavior priors, answer actions | Trusted machinery |
| Exact posterior identity and telescoping implementation | Trusted machinery |
| Historical corrected 3B KL-fixed metrics | Historical baseline only |
| Legacy support/strict metrics | Invalid for new claims or rewards |
| Local v2 support teacher | Rejected; forbidden downstream |
| v3 structured teacher pilot | Trusted pilot GO |
| Full v3 support labels | In progress; not yet final |
| Runtime no-gold support verifier | Planned after full labels |
| Corrected evidence/query/visual/full-DAG selector result | Not yet available |
| Learned policy / GRPO / GDPO from corrected DAG-IG | Not started |
| Top-conference paper claim | Not established |

## Current Blocker

The immediate blocker is not missing Serper quota or GPU capacity. It is a
scientific contract: a runtime no-gold verifier and terminal-success value must
be calibrated against trustworthy semantic-support labels before exact
backward credit can be recomputed. Reusing the old support label would make the
method appear complete while invalidating the core claim.

No new real-search calls are needed for the current 2,954 query actions. API
secrets, private labels, request logs, raw examples containing gold data, model
weights, and local caches are intentionally excluded from this repository.

## Required Next Steps

1. Finish the four resumable full-v3 label shards and verify exact 12,365-state
   and 14,770-action coverage.
2. Aggregate physically separate policy-train and internal private labels.
3. Fit a no-gold runtime support verifier on policy-train only; freeze internal
   predictions before opening internal labels.
4. Recalibrate answer-action terminal success from corrected support and frozen
   reader confidence.
5. Recompute answer and evidence posteriors, then compare No-credit, Local-IG-M,
   Outcome, and DAG-IG under identical actions and budgets.
6. If evidence passes, proceed backward once through query, visual, and root;
   audit posterior identities, telescoping, leakage, and paired gains at each
   node.
7. Freeze the complete selector-only method on the 158/40 development split.
8. Only after full-DAG selector GO, expand to at least 1,000 unique training
   tasks and a harder frozen corpus. Multiple rollouts do not substitute for
   unique tasks.
9. Open dev once and test once after method/data freeze; run three seeds and
   paired bootstrap/randomization significance.
10. Add an official external multimodal-search benchmark and compare with
    external systems only under a matching search/corpus/tool protocol.
11. Attempt scorer distillation or bounded Node-GDPO/GRPO only as a later
    extension. Query-only DPO is not the main method.

## Repository Map

- Current roadmap: `v6/reports/DAGIG_V6_TOP_CONFERENCE_RESULT_ROADMAP.md`
- v6 code snapshot: `v6/scripts/` and `v6/dagig_causal/`
- v6 method specs: `v6/specs/`
- Key audits: `v6/reports/`
- Historical 3B code/results: `scripts/`, `results/`, and `paper/`
- Model/data release boundary: `MODEL_AND_DATA.md`

## Claim Discipline

The correct current statement is:

> DAG-IG now has an executable counterfactual DAG and an auditable exact
> posterior-credit implementation. A provisional evidence-selector signal was
> found, but the legacy support label was invalid. A structured cited semantic
> label contract passed an independent pilot, and the full corrected
> selector-only experiment is in progress. A top-conference-level empirical
> result has not yet been established.
