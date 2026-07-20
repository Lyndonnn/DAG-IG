# Cached Multi-Query Evidence v2 Status

## Scope

This stage used only complete cached real-search query states. It made zero API calls, did not train a generator, and did not open dev or test.

## Frozen Universe

- Samples: 198
- Query states: 1,184
- Policy-train/internal states: 946 / 238
- Evidence actions: 5,920
- Actions per state: five fixed A-E interventions
- Exclusions: two cached states with fewer than five results and two states absent from the complete frozen no-gold action/value universe
- DAG posterior identity maximum error: 0.0

Public target files contain only `parent_state_id`, `prompt`, five legal actions, and the No-credit/Local-IG/Outcome/DAG-IG target distributions. Gold, qrels, support, strict, answer correctness, and terminal values are absent.

## Selector-Only Result

The preregistered direct posterior selector passed all gates.

| Method | Expected terminal | Support | Expected strict | Mode strict |
|---|---:|---:|---:|---:|
| No-credit | 0.154225 | 24.79% | 13.48% | 14.71% |
| Local-IG | 0.197051 | 28.99% | 15.84% | 17.23% |
| Outcome | 0.194084 | 29.41% | 15.48% | 16.39% |
| DAG-IG | **0.199783** | 28.57% | **16.10%** | **17.23%** |

DAG-IG versus No-credit expected-terminal delta is +0.045558, with sample-clustered 95% CI `[+0.035356, +0.057185]`. DAG-IG versus Outcome delta is +0.005698, with CI `[+0.003312, +0.008614]`. DAG-IG differs from Outcome on 26.47% of top actions and uses all five evidence strategies.

Decision: `DAGIG_V6_CACHED_MULTIQUERY_SELECTOR_ONLY_GO`.

## Scalar Ranker Result

Four BGE state-action scalar rankers were trained for three epochs with listwise KL. The learned projection did not reproduce the posterior ordering.

| Method | Internal TV | Internal top agreement | Train TV | Train top agreement |
|---|---:|---:|---:|---:|
| No-credit | 0.003833 | 21.43% (uniform tie) | 0.004017 | 19.45% (uniform tie) |
| Local-IG | 0.136911 | 26.89% | 0.131225 | 24.74% |
| Outcome | 0.167331 | 22.69% | 0.165540 | 22.73% |
| DAG-IG | 0.134293 | 23.53% | 0.130438 | 24.95% |

The DAG ranker does improve expected terminal value over the learned No-credit ranker by +0.011257, but it fails the frozen top-agreement gates. Because train agreement is also near random, this is a target-fit/representation-objective failure, not primarily an internal generalization failure.

Decision: `DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_NO_GO`.

## Protocol Conformance Finding

The v2 trainer set the random seed after creating the LoRA adapter. Therefore, the code did not guarantee an identical random LoRA initializer across the four methods, despite the freeze claiming matched initialization. Cross-method ranker comparisons are not paper-valid. The per-method train-fit failure remains diagnostic, because every non-uniform method fails to fit its own target.

Do not rerun the same internal split to tune this ranker. If a scorer is revisited, it must receive a new protocol version, set the seed before adapter construction, and be selected using policy-train-only grouped cross-validation. The original internal result must not be reused for hyperparameter selection.

## Erratum

`cached_multiquery_ranker_v2_audit/DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_REPORT.md` contains an incorrect final recommendation. Its report writer tested `decision.endswith("_GO")`, which is also true for a string ending in `_NO_GO`. The authoritative JSON decision and this status report supersede that sentence. No ranker is approved for downstream use.

## Current Method Decision

The clean result supports the DAG-IG credit formulation at the direct selector level: exact backward value aggregation selects higher-value evidence than No-credit, Outcome, and Local-IG under the frozen verifier and shared answer policy. It does not yet show that this posterior can be distilled into the tested scalar ranker.

Keep the direct posterior selector as the current executable evidence-node method. Do not open dev/test, train another categorical generator, or escalate epochs from this result.
