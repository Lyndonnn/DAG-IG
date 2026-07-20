# DAG-IG v3 Method Specification

> Historical, superseded protocol. It is retained only to reproduce legacy
> diagnostics; `METHOD_SPEC.md` and `TRAINING_SPEC.md` are authoritative.

## Status

This is the replacement design for the paper main method. It preserves the
repaired immutable-state and frozen-value work from Causal v2.1, replaces the
bare-rank evidence action with the v2.2 listwise pointer, and fixes the credit
estimator used by the optimizer. It is not a result claim or a protocol freeze.

## Executable DAG

```text
image, question
  -> visual_action = {grounding_expression, visual_anchor}
  -> frozen GroundingDINO proposals/crops
  -> search_query
  -> frozen BM25 top-20
  -> evidence_action = unordered set of three candidate labels
  -> final_answer
```

All policy actions are generated without gold answers, qrels, teacher queries,
or oracle crops. Gold answers and support labels are available only to the
train-time frozen value/verifier and to sealed evaluation.

## Information Potential on the DAG

Let `S_0, ..., S_3` be the executable acquisition states before visual action,
after visual grounding, after retrieval, and after evidence selection. A frozen
success estimator defines a log potential `V_phi(S)`. The realized edge
information gain is

```text
Delta I_n = V_phi(S_{n+1}) - V_phi(S_n).
```

The sum telescopes exactly to `V_phi(S_3) - V_phi(S_0)`. Local edge credit uses
only `Delta I_n` and is a required control. The main method instead assigns an
action its expected cumulative downstream information gain under a surgical
intervention. For a fixed parent state, subtracting `V_phi(S_n)` is constant,
so its policy gradient is determined by the counterfactual terminal acquisition
potential defined below. This is the causal-credit component of DAG-IG. The
answer is the sink and receives separate verified utility credit because it is
an emitted decision, not another information-acquisition state.

## Same-Parent Counterfactual Credit

For an actual action `a_n^(0)` at node `n` with parent state `s_n`, draw `K`
counterfactual actions independently from the unchanged behavior policy.  The
actual draw and counterfactual draws form one immutable policy group:

```text
G_n(s_n) = {a_n^(0), ..., a_n^(K)}
a_n^(k) ~ pi_old(. | s_n),  k = 0..K
```

No-op draws and duplicate draws retain their Monte Carlo mass. For every draw,
replace only node `n` and replay every descendant with common descendant random
seeds. The question, image, corpus, model snapshot, tool configuration, and all
parents of `n` are immutable. Any no-op must produce byte-equivalent canonical
value context and exactly zero *pairwise causal delta*. Duplicate no-op draws
must receive identical `Q`, `IG`, and optimizer advantage; that shared groupwise
advantage need not be zero when other sampled actions in the group differ.

For upstream nodes (`visual_action`, `search_query`, `evidence_action`), let

```text
ell_phi(tau) = log p_phi(gold_answer | question, selected evidence)
```

where `phi` is a frozen evidence reader and `L` is the number of gold-answer
tokens. Every candidate action is evaluated by replaying all descendants with
common random numbers. The action value is a normalized log marginal success
probability, not an expectation of log probability:

```text
Q_n(s_n, a_n^(i))
  = (1/L) log E_desc[exp ell_phi(tau(s_n, a_n^(i), desc))].
```

The raw, reportable DAG information gain of each sampled action is

```text
IG_n(s_n, a_n^(i))
  = Q_n(s_n, a_n^(i))
    - (1/L) logmeanexp_j(L Q_n(s_n, a_n^(j))).
```

This order of operations is required: taking `E[log p]` or applying
`logmeanexp` to per-token means is not the marginal success probability. The
result is a length-normalized per-example log Bayes factor / pointwise
information-gain estimate and satisfies
`logmeanexp_j(L * IG_n^(j)) = 0` exactly up to numerical precision. Structured
corruptions are diagnostics only and never enter the main estimator.

The optimizer must not train only the actual action against the counterfactual
baseline.  `logmeanexp` has a finite-sample negative offset, so actual-only
records can move every sampled action downward.  Instead, all `K+1` actions in
the same-parent group are policy samples.  Their Node-GRPO advantage is the raw
IG with a same-parent control variate:

```text
A_n^(i) = IG_n^(i) - mean_j IG_n^(j)
        = Q_n^(i) - mean_j Q_n^(j).
```

Thus each policy group has exactly zero-sum advantages.  This control variate
does not change the expected policy gradient and prevents cross-state reward
offsets from becoming a spurious update direction.

This also gives an explicit baseline-invariance sanity theorem: a
"same-parent terminal-Q" objective built from exactly the same interventions
is algebraically identical after group centering. It is not presented as an
independent empirical baseline. The meaningful terminal-reward control samples
root trajectories and copies one trajectory outcome to every reached node;
unlike DAG-IG, it does not perform same-parent interventions at each node.

The answer node is the sink. Its credit uses the same-parent verified utility
difference:

```text
Q_answer(s_A, a_A^(i)) = U(a_A^(i), evidence),
C_answer^(i) = Q_answer^(i) - mean_j Q_answer^(j),
```

where `U` is frozen before optimization and combines answer correctness,
evidence support, format validity, and answer-in-query leakage penalties. The
answer counterfactuals are sampled from `pi_old(. | s_A)`; no gold answer is put
in the answer-policy prompt. `C_answer` is terminal verified utility credit, not
mislabelled as upstream information gain.

## Variance Control

1. Use at least `K=4` policy-marginal draws in the main run. Report a `K=2/4/8`
   stability diagnostic on a train/dev subset.
2. Reuse common random numbers for all descendants of an actual/CF pair.
3. Evaluate all actions in one policy group under the same descendant seeds.
   If exactly the same canonical action is repeated in that group, replace its
   value estimates by their mean before centering.  In the main run, repeated
   descendant seeds provide Rao-Blackwellization over descendant randomness.
4. Center only within the exact same-parent policy group.  Never z-score or
   center actions from different parent states.
5. Apply one train-fitted robust scale per node.  If clipping is needed, rescale
   the complete policy group symmetrically so zero-sum advantages are preserved.
   Freeze these statistics before dev/test evaluation.

## Policy Objective

For semantic action tokens (or the exact evidence-set probability), optimize

```text
L_n = -E[min(r_n A_n, clip(r_n, 1-eps, 1+eps) A_n)]
      + beta * KL(pi_theta || pi_old).
```

The old and current policy must be identical before the first update: adapter
dropout is disabled for both cached and current log probabilities. Complete
state groups are not split across optimizer steps. Evidence uses an exact
unordered top-3 Plackett-Luce set probability over state-local labels A--T; bare
rank JSON is forbidden in the main method.

The default paper implementation is node-factorized LoRA with one adapter per
policy node. Every reward baseline receives the same four-adapter parameter
budget, rollout budget, prompts, tools, and optimizer settings. Adapter switching
is part of the agent runtime and is fixed across methods.

## Required Controls

1. Format-SFT behavior policy without RL updates.
2. Trajectory-level terminal reward copied to every reached acquisition node.
3. Query-only information gain.
4. Local/non-descendant edge credit.
5. Full DAG-IG.
6. Full DAG-IG ablations without visual, evidence, or answer counterfactual
   credit.

The algebraically equivalent same-parent terminal-Q formulation is checked by
an exact record-level identity test rather than trained as a duplicate model.

All controls must rank the same candidate universe and use identical frozen
tools. DPO is a pairwise ablation, not the main optimizer.

## Gates Before Main Training

1. Immutable-state, no-op identity, and replay-completeness gates pass.
2. Credit direction predicts controlled descendant value changes on held-out
   dev interventions with sample-clustered confidence intervals.
3. Positive and negative one-record optimizer sign tests start at ratio `1.0`
   and KL `0.0` for every node.
4. Full train-only objective audit passes both the sample-clustered
   `A * delta log pi` gate and the same-parent pairwise relative-log-odds gate
   `(A_i-A_j) * delta(log pi_i-log pi_j)`, with no policy group split and no
   test rows. Absolute positive/negative probability movement is reported only
   as a diagnostic because softmax mass can move to unsampled actions.
5. A 418-task controlled diagnostic shows a full-method advantage over the
   terminal-only and query-only controls before any API-backed expansion.

Only after these gates pass may the project spend Serper/OpenAI quota, unlock
three-seed runs, evaluate test, or run the 7B main experiment.
