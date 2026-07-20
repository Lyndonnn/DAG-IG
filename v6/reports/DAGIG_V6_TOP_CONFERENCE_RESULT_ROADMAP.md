# DAG-IG v6 Top-Conference Result Roadmap

Date: 2026-07-20

## Decision

The current system is not a paper-final result. The executable counterfactual
DAG is credible, but the legacy evidence-support label contract is invalid.
Consequently, the previous evidence/query/visual support and strict-success
GO/NO-GO results are provisional and must be recomputed after label repair.

Do not train an actor, run RL, open frozen dev/test, or claim the previous
evidence GO until the new semantic-support labels pass an independent audit.

## Frozen DAG-IG Method

The trajectory remains:

```text
image + question
  -> visual action (OCR / caption / joint)
  -> structured query action (3-5 real-search interventions)
  -> cached real search
  -> evidence action (five A-E evidence-set interventions)
  -> frozen shared answer policy
  -> final answer
```

For success event `Y`, the intended node credit is the exact
control-as-inference posterior:

```text
q_n(a | s, Y=1) = mu_n(a | s) V_{n+1}(s,a) / V_n(s)
IG_n(s,a) = log V_{n+1}(s,a) - log V_n(s)
```

The edge credits telescope along a path. No-credit, Local-IG-M, Outcome, and
DAG-IG must receive identical legal actions, downstream policies, search calls,
reader, and KL/compute budgets.

## What Is Currently Trustworthy

- 198 train-development samples are frozen as 158 policy-train and 40 internal
  method-development samples; official dev/test remain sealed.
- 2,954 complete query actions use cached real search, with no new search calls
  required for current method development.
- 14,770 executable evidence actions preserve five legal interventions for each
  query state.
- A shared frozen answer policy has 41,273 answer actions.
- Behavior priors, posterior normalization, information identity, and path
  telescoping are implemented and auditable.
- Runtime action records contain no gold answer, qrel, target document, support
  label, or final-correctness feature.

These facts establish the experimental machinery. They do not establish that
DAG-IG improves task success.

## Invalidated Support Contract

The legacy support label was:

```text
positive URL match
OR
normalized gold-answer phrase occurs anywhere in the selected titles/snippets
```

It did not check entity identity, location/time constraints, or whether a short
number occurred in the correct factual context. On 11,795 policy-train evidence
actions:

| Legacy trigger | Actions |
|---|---:|
| Positive URL and answer phrase | 991 |
| Positive URL only | 663 |
| Answer phrase only | 1,473 |
| Negative | 8,668 |

Among the answer-phrase-only positives, 380 have short numeric answers. Manual
inspection found both directions of label error:

- unrelated evidence containing an incidental `2`, `3`, or `18` was positive;
- semantically equivalent addresses and transliterations were negative;
- correct values requiring question-requested rounding were often negative.

The audit decision is `DAGIG_V6_SUPPORT_LABEL_CONTRACT_INVALID`. The previous
evidence-node result (29.92% support / 15.73% expected strict) and all later
support comparisons are retained only as historical diagnostics, not as valid
paper evidence.

## Runtime Verifier Result

An answer-independent frozen Qwen2.5-VL-7B verifier was scored on all 2,954
query states using only:

```text
question + visual observation + executed query + selected evidence
```

It used one-token `A/B` probabilities and no gold/private fields. Policy-train
grouped OOF produced:

| Metric | Result | Frozen gate |
|---|---:|---:|
| Support AUC against legacy label | 0.7166 | >= 0.80 |
| Brier improvement | -0.0007 | >= 0.01 |
| Within-visual pair order | 0.7493 | >= 0.68 |
| Nonconstant groups | 100% | >= 95% |

Decision: strict NO-GO; internal query labels were not opened. The strong pair
ordering but weak global calibration was the signal that exposed the legacy
label problem rather than a reason to relax the gates.

## Semantic Label Repair

The local Qwen next-token v2 teacher failed its frozen independent audit. On 350
blinded policy-train actions it achieved only 65.43% balanced accuracy, 79.75%
precision, and 56.25% recall against an independent GPT-5-mini reference. Its
continuous score AUC was 0.7944, but even the best diagnostic threshold reached
only 0.7376 balanced accuracy. The failure included 87 false negatives where an
exact or normalized answer was visibly present in a selected snippet. V2 labels
remain forbidden for training, value construction, and claims.

The replacement v3 contract uses a structured cited semantic teacher. Every
positive label must name a supporting document and provide a span verifiable
against that document. A fresh 400-state pilot, disjoint from the v2 audit in
both action ID and `(sample, selected docs)` state, used GPT-5.4-mini as teacher
and a blinded GPT-5.4 auditor. It passed every pre-registered gate:

| Metric | v3 pilot |
|---|---:|
| Accuracy | 97.50% |
| Balanced accuracy | 97.17% |
| Precision | 92.39% |
| Recall | 96.59% |
| Disagreements | 10 / 400 |

All numeric, phone/identifier, email, and address subset gates passed, as did
teacher/auditor citation validity. Dev/test and Serper were not used.

The full label protocol is now frozen. The 14,770 evidence actions deduplicate
to 12,365 semantic states; 400 audited teacher states are reused and the
remaining 11,965 are being scored in four resumable shards. These private labels
are evaluation/value supervision only. Runtime selectors cannot read gold,
private support labels, or answer correctness.

## Required Next Steps

### 1. Finish full v3 labels and the no-gold verifier

Complete all four resumable v3 label shards and require exact coverage of the
12,365-state/14,770-action frozen universe. Then train/calibrate a runtime
semantic support verifier on policy-train labels only. Freeze its predictions
before opening internal labels. The verifier must use only question, visual
observation, executed query, and selected evidence.

### 2. Recompute the terminal and evidence nodes from scratch

After label GO, rebuild:

```text
V_support(e) = calibrated runtime semantic support probability
V_strict(e)  = V_support(e) * P(answer correct | supported evidence)
```

Re-evaluate all five evidence interventions for No-credit, Local-IG-M, Outcome,
and DAG-IG. The evidence node is GO only if DAG-IG improves/non-inferior strict
success and support under the corrected labels. Previous evidence models may be
used as baselines, not silently carried forward as trusted values.

### 3. Recompute query, visual, and full-DAG values

Proceed backward only after the child node passes:

```text
answer -> evidence -> query -> visual -> root
```

At every node:

- execute every legal action through the frozen downstream controller;
- form `V_n`, posterior `q_n`, and `IG_n` from cardinal probabilities;
- audit exact posterior identity and telescoping;
- compare equal-budget No-credit, Local-IG-M, Outcome, and DAG-IG;
- stop on a failed node instead of moving failure upstream.

Full-DAG direct-controller GO requires corrected support and strict success to
beat or be noninferior to both Local-IG-M and Outcome, distributed paired gains,
action diversity, and no answer leakage.

### 4. Complete selector-only before any distillation

The immediate paper result is the complete direct DAG-IG posterior selector.
Policy/ranker distillation is deferred because it has repeatedly obscured the
credit-assignment test. First prove the full selector under equal legal actions,
search results, evidence sets, answer policy, and compute. Only after the
selector result is frozen may Node-GDPO/grouped GRPO be attempted as an optional
extension; query-only DPO is not the main method.

### 5. Scale after method GO

The current 158/40 split is for method development, not the final empirical
claim. After full-DAG direct GO, construct at least 1,000 unique training tasks
from Pix2Fact-WebEvidence and official MMSearch-Plus training resources. Multiple
rollouts from one image do not replace unique tasks.

No new Serper calls are needed for the current 2,954 states. Preserve the unused
`SERPER_API_KEY_EVAL` quota for the final frozen external evaluation. Added data
must use official images/questions, cached source pages, automatic legal node
interventions, source/entity decontamination, and a small independently audited
support subset.

### 6. Sealed paper evaluation

After method/data/training freeze:

- open dev once, then test once;
- run three seeds and paired bootstrap/randomization significance;
- compare Base, Format-SFT, Outcome-GRPO, Local-IG-M, DAG-IG, and DPO ablation;
- report strict success, answer correctness, corrected support, retrieval,
  leakage, cost, and node/path credit calibration;
- run 3B and 7B scaling;
- evaluate on an official external multimodal-search split and compare with
  MMSearch-R1-7B only under a matching protocol.

## Stop Conditions

Do not:

- use legacy support/strict labels for new claims or rewards;
- enable provisional v2 labels before independent audit GO;
- open frozen dev/test;
- spend new search quota on already cached states;
- train GRPO/GDPO before full-DAG direct-controller GO;
- relax failed gates after observing a partition;
- use teacher/oracle/gold fields in runtime scorers;
- return to query-only DPO as the main DAG-IG method.

## Key Artifacts

- Legacy label contract audit:
  `/root/dagig_scratch/v6_full_dag/support_label_contract_audit_v1_fixed/`
- Runtime semantic verifier freeze/scores/train NO-GO:
  `/root/dagig_scratch/v6_full_dag/semantic_support_verifier_protocol_v1_fixed/`
  `/root/dagig_scratch/v6_full_dag/semantic_support_verifier_scores_v1_fixed_shard0/`
  `/root/dagig_scratch/v6_full_dag/semantic_support_verifier_scores_v1_fixed_shard1/`
  `/root/dagig_scratch/v6_full_dag/semantic_support_verifier_train_oof_v1_fixed/`
- Gold-aware support teacher v2:
  `/root/dagig_scratch/v6_full_dag/gold_aware_support_teacher_protocol_v2_fixed/`
  `/root/dagig_scratch/v6_full_dag/gold_aware_support_teacher_scores_v2_fixed_shard0/`
  `/root/dagig_scratch/v6_full_dag/gold_aware_support_teacher_scores_v2_fixed_shard1/`
- Provisional labels and blinded independent audit pack:
  `/root/dagig_scratch/v6_full_dag/gold_aware_support_labels_v2_fixed/`
- Failed v2 independent audit and root cause:
  `/root/dagig_scratch/v6_full_dag/gpt_support_label_audit_v1_fixed2/`
  `/root/dagig_scratch/v6_full_dag/support_label_v2_root_cause_v1/`
- Structured cited v3 pilot GO:
  `/root/dagig_scratch/v6_full_dag/structured_support_teacher_pilot_v3_fixed3_audit/`
- Frozen full v3 label protocol:
  `/root/dagig_scratch/v6_full_dag/structured_support_labels_full_v3_protocol_fixed6/`

## Bottom Line

The DAG-IG main line is intact, but current performance claims are not yet final.
The semantic label bottleneck has progressed from an invalid legacy proxy and a
failed v2 teacher to a cited v3 contract that passed a fresh 400-state blinded
audit. The shortest credible route is now: finish full v3 labels, calibrate the
no-gold verifier, recompute every node backward, prove the full-DAG direct
selector, then scale data and run sealed external evaluation. Policy
distillation is not required for the current selector-first paper result.
