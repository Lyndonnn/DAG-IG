# DAG-IG Method Specification

Date: 2026-07-16

## 1. Claim

DAG-IG is a training-time causal credit estimator for multimodal search
agents. It measures how much a policy action at one executable DAG node changes
the probability of downstream supported success when that node is intervened
on and all affected descendants are re-executed.

The optimizer is not the contribution. The contribution is the node-local
counterfactual transition protocol, frozen supported-success value, and
probability-ratio information credit. At inference the policy runs once with
no gold answer, qrels, support labels, counterfactuals, teacher query, or path
reranking.

## 2. Agent DAG

```text
X = image + question
  -> A_v  visual observation action
  -> A_q  search-query action
  -> O_r  frozen search/retrieval observation
  -> A_e  evidence-selection action
  -> A_a  final-answer action
  -> Y    correct-and-supported success
```

`A_v`, `A_q`, `A_e`, and `A_a` are trainable policy nodes. `O_r` is an
environment node. The visual action is an observable description/anchor; a
mandatory crop or GroundingDINO route is not part of the main claim. Grounding
can be evaluated as an auxiliary action or diagnostic. To prevent a hidden
causal bypass, only `A_v` reads the raw image in the main four-node system;
query, evidence, and answer stages receive the frozen visual observation and
their other declared parents, not the raw image. An image-visible reader is
allowed only as a separately labeled query-node sensitivity diagnostic.

Canonical stage outputs are compact and separately generated:

```json
{"visual_observation":"..."}
{"search_query":"..."}
{"selected_doc_ids":["d1","d2","d3"]}
{"final_answer":"..."}
```

## 3. One-Node Intervention

At node `i`, freeze its realized parent state `s_i`, sample an action from the
unchanged reference policy, and re-execute every affected descendant:

```text
s_i -> do(A_i=a_i) -> descendants under frozen policies/environment -> Y.
```

Main counterfactual actions are iid draws with replacement from the same
reference node policy. Duplicate/no-op actions keep their probability mass.
Structured corruptions are diagnostics only. Replacing text without rerunning
retrieval, evidence selection, and answering is not a valid upstream
counterfactual.

## 4. Supported-Success Value

For node action `a_i`:

```text
V_i(a_i,s_i)
  = P_frozen(Y=1 | s_i, do(A_i=a_i)).
```

`Y=1` requires both answer correctness and direct evidence support. The value
teacher is frozen before policy optimization and identical across actions and
methods. Train answers may supervise this teacher after rollout; they never
enter policy inputs. Dev/test answers are excluded from reward fitting.

The current query-node mechanism pilot estimates `V` from the full terminated
gold-answer sequence likelihood of one fixed reader and a grounded support
factor. A calibrated frozen verifier is allowed only after an independent
faithfulness audit. Retrieval rank or answer EM alone is not `V`.

## 5. Causal Pointwise Information Gain

Let `w_k` be the reference Monte Carlo mass represented by action draw `k`.
The weighting rule is fixed by acquisition, not tuned from outcomes:

- iid draws from `pi_ref`: `w_k=1/K`, with duplicates retained;
- exhaustive finite action enumeration: `w_k=pi_ref(a_k|s)` normalized on the
  enumerated support;
- draws from a known proposal `mu`: `w_k` is proportional to
  `pi_ref(a_k|s)/mu(a_k|s)`.

Then:

```text
B_i(s) = logsumexp_k(log w_k + log V_i(a_k,s)).

IG_i(a_k,s) = log V_i(a_k,s) - B_i(s).
```

The supported-success posterior is:

```text
p_i(k|Y=1,s) = w_k * exp(IG_i(a_k,s)).
```

Therefore:

```text
sum_k w_k * exp(IG_i(a_k,s)) = 1

E_{p_i(.|Y=1,s)}[IG_i]
  = KL(p_i(.|Y=1,s) || w(.|s)) >= 0.
```

These executable identities distinguish DAG-IG from a renamed score delta.
An action can have positive or negative pointwise IG; the group's conditional
expected information is nonnegative.

When `V` is a calibrated binary success probability (clipped only for numeric
logs to `[1e-6,1-1e-6]`), DAG-IG also reports the
full Shannon information between the node action and outcome. Let
`p=sum_k w_k V_k`:

```text
I(A_i;Y|s_i)
  = sum_k w_k [
      V_k     * log(V_k/p)
      + (1-V_k) * log((1-V_k)/(1-p))
    ] >= 0.
```

`IG_i(a,Y=1)=log V_i(a)-log p` is the success-directed pointwise credit used
for policy improvement. Full binary mutual information is a mechanism metric,
not the optimization reward, because an action that reliably predicts failure
can be informative but should not be promoted.

The credit is node-local and conditioned on the realized parent state. The
method does not assume that arbitrary samplewise node credits add without
interaction. Parent conditioning and descendant re-execution capture
downstream interactions; environment observations are reported separately.

The current v5.5 mixed-proposal pilot has no known proposal density. It reports
an explicitly nonfinal finite-support projection using normalized exact
reference likelihoods. A final claim requires iid reference-policy draws and
uniform per-draw Monte Carlo mass; exact action log probabilities are then used
for policy gradients/KL diagnostics, not to double-weight the baseline. The
pilot's uncalibrated support potential also cannot be used to report full
binary mutual information.

## 6. Node Semantics

### Visual

Hold image/question fixed. Resample a concise visual observation, then rerun
query, search, evidence selection, and answer. Reject newly introduced answer
leakage.

### Query

Hold image/question/visual state fixed. Resample the query, execute real or
frozen-version search, then rerun evidence selection and answer.

### Evidence

Hold the retrieved candidate pool and upstream state fixed. Resample exactly
the allowed evidence subset, then rerun the answer. Candidate order must be
randomized or modeled permutation-invariantly.

### Answer

Hold image/question/evidence fixed. Resample the concise answer action and use
a frozen support/correctness verifier. Negatives must be plausible matched
hypotheses, not unrelated length-matched strings.

## 7. Optimization

The Bayesian target for node `i` is:

```text
p_beta(k|s) proportional to w_k * exp(beta * IG_i(a_k,s)).
```

The main optimizer is node-masked GDPO over this target with a KL trust region.
A group of iid `pi_old` draws uses the finite-group current policy

```text
p_theta^K(k|s) proportional to
  w_k * exp(log pi_theta(a_k|s) - log pi_old(a_k|s)).
```

Thus `p_theta^K=w` at initialization. The old-policy subtraction is required:
the iid draw frequency already represents `pi_old`, so applying current
sequence likelihood without the ratio would double-count policy mass.
A node-masked GRPO objective is a required ablation. Only tokens generated by
the credited node receive its loss. DPO may distill a passed target but is not
the main validation.

## 8. Required Baselines

- format/query SFT;
- outcome GRPO;
- trajectory GRPO;
- query-only IG-Search-style GRPO;
- Local-IG without descendant replay;
- DAG-IG query-only;
- full four-node DAG-IG;
- leave-one-node-out DAG-IG;
- optional DAG-IG DPO distillation.

Every method uses the same model initializer, rollout parents, action/search
budget, evidence budget, reader, optimizer-step budget, and seeds.

## 9. Invalid Claims

DAG-IG is not validated by any of the following alone:

- a scorer correlates with labels;
- an oracle selector has headroom;
- DPO beats one SFT checkpoint;
- query credit works while other claimed nodes are placeholders;
- descendants are not re-executed after intervention;
- results use mixed teacher/oracle provenance;
- one small split improves without uncertainty or external confirmation.

The final claim requires a no-label trained policy, causal-credit audits,
node ablations, fair GRPO/GDPO baselines, 3B/7B scaling, and external validity.
