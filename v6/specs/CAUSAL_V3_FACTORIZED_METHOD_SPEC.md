# DAG-IG Causal v3: Factorized Node Policies

> Historical, superseded protocol. It is retained only to reproduce legacy
> diagnostics; `METHOD_SPEC.md` and `TRAINING_SPEC.md` are authoritative.

## Motivation

Causal v2 established valid counterfactual node credit, but a shared LoRA did
not preserve all node advantages. Query-only IG improved retrieval while full
DAG-IG improved evidence support; combining them in one adapter caused
cross-node interference. A sampled train audit measured only 58.6% update
direction agreement, and node-homogeneous accumulation did not resolve
interference across successive node updates.

## Policy

The executable DAG and all rewards remain unchanged:

```text
image + question
  -> visual policy head
  -> frozen GroundingDINO proposals/crops
  -> query policy head
  -> frozen BM25 top-20
  -> evidence policy head
  -> final-answer policy head
```

All four heads share the same frozen Qwen2.5-VL backbone and Format-SFT
initializer, but each has its own LoRA parameters. At inference, the agent
activates the matching adapter before each stage call.

## Credit and Optimization

The pointwise information-gain estimator, terminal reward, alpha, clipping,
and token masks are identical to Causal v2. Each head receives only its own
advantage:

```text
visual/query/evidence: z(R_terminal) + 0.5 * clip(z(IG_node), -3, 3)
final answer:          z(R_terminal)
```

For the first controlled pilot, every method has four LoRA heads and 100 total
optimizer steps: 25 steps per head, accumulation 8, learning rate 1e-6. The
trajectory, query-only, and full DAG-IG controls differ only in the advantages
used to build each head's records.

## Gate

Selection uses deterministic evaluation on the same 42 dev samples. Test is
sealed. Full factorized DAG-IG advances only if strict success exceeds both
factorized controls without an R@5 regression. If this gate fails, the
199-sample experiment stops; no further reward, schedule, or optimizer changes
are allowed before revisiting data scale or the core credit hypothesis.
