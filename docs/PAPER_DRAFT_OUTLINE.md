# DAG-IG Paper Draft Outline

## Working Title

DAG-IG: Node-Level Credit Assignment for Long-Horizon Multimodal Search Agents

## Core Thesis

Long-horizon multimodal search agents fail because final answer reward is too sparse to identify whether the visual observation, search query, retrieved evidence, or answer extraction caused success or failure. DAG-IG assigns node-level credit to these stages and uses the resulting reward to optimize the stage-1 policy of a two-stage Pix2Fact agent.

The current paper result should be stated narrowly: under a frozen offline Pix2Fact evidence-note BM25 evaluation, KL-fixed DAG-IG GRPO improves a Format-SFT two-stage agent on dev and test as a two-seed mean, and the training rewards are non-collapsed. It should not be phrased as a causal counterfactual-intervention result.

## Contributions

1. A DAG-structured rollout formulation for multimodal fact-seeking agents:
   `image + question -> visual_observation -> search_query -> retrieve top-k evidence -> final_answer`.
2. A node-level DAG-IG reward that separates visual, query, evidence, answer, format, and leakage terms instead of relying only on final answer correctness.
3. A two-stage GRPO training setup that optimizes the stage-1 policy while keeping retrieval and reader evaluation fixed and auditable.
4. Empirical evidence on Pix2Fact showing a modest two-seed improvement over Format-SFT, plus a fixed-reader control.
5. A reward/training-health audit showing low constant-reward group rate after the k3 KL fix.

## Main Method Section

### Agent Rollout

Define the rollout as:

```text
x: image, question
z_v: visual_observation
z_q: search_query
z_e: retrieved top-k evidence
y: final_answer
```

The policy emits only `z_v` and `z_q` in stage 1. Retrieval is deterministic BM25 over the frozen corpus. The reader receives image, question, and top-5 evidence and emits `final_answer`.

### DAG-IG Credit

Describe each reward component:

- visual credit: whether visual observation preserves useful anchor/entity information.
- query credit: retrieval quality of the generated query.
- evidence credit: whether top-k retrieved evidence contains support.
- answer credit: final answer correctness under evidence support.
- format credit: compact valid JSON and nonempty fields.
- leakage/path penalties: discourage answer-in-query and degenerate traces.

Emphasize that reward ranking is group-relative inside GRPO and audited post-hoc.

### Optimization

Use grouped GRPO with:

- base: Qwen2.5-VL-3B-Instruct
- initializer: Format-SFT
- stage1-only policy loss
- KL=0.1 with a non-negative k3 KL estimator
- num generations=4
- top-k retrieval=5
- 60 optimizer steps for the main run

## Experimental Setup

### Dataset

Use the clean Pix2Fact diagnostic setup. Keep the description high-level:

- image-question samples with offline evidence corpus
- dev/test use frozen BM25 evaluation
- no real web search
- no teacher/oracle query in evaluation

Mention that old unclean data and oracle trajectories are excluded from the paper-main training/evaluation path. Also state that the corpus is a frozen Pix2Fact evidence-note corpus, not live web or noisy full documents.

### Baselines And Controls

Main table should include:

- Format-SFT
- KL-fixed DAG-IG seed42
- KL-fixed DAG-IG seed43
- KL-fixed two-seed mean

Older experiments should be appendix/diagnostics only:

- DAG-SFT trace imitation
- outcome-only SFT smoke/full pilots
- DPO/pair scoring attempts
- query reranking/switching
- multi-query fusion
- broad answer repair
- query-warm GRPO variants

These are useful for motivating design decisions but should not be framed as the paper main method.

## Main Results

Use `outputs/dagig_paper_main_v1/paper_assets/main_results_table.tex`.

Key numbers:

| method | dev R@5 | dev strict | test R@5 | test strict |
|---|---:|---:|---:|---:|
| Format-SFT v4 | 52.0% | 40.8% | 46.9% | 34.4% |
| KL-fixed GRPO seed42 | 56.1% | 45.9% | 51.6% | 40.6% |
| KL-fixed GRPO seed43 | 56.1% | 45.9% | 48.4% | 37.5% |
| KL-fixed GRPO two-seed mean | 56.1% | 45.9% | 50.0% | 39.1% |

Paper wording:

KL-fixed DAG-IG improves strict success over Format-SFT by +5.1 dev points and +4.7 test points as a two-seed mean. Seed42 individually reaches 40.6% test strict and seed43 reaches 37.5%, so the clean headline is the two-seed mean rather than best test seed.

## Reward And Credit Diagnostics

Use `outputs/dagig_paper_main_v1/paper_assets/node_credit_diagnostic_table.tex`.

Main claims:

- KL-fixed seed42 has 3/240 constant-reward groups.
- KL-fixed seed43 has 1/240 constant-reward groups.
- the earlier 78.8% constant-reward concern does not apply to the KL-fixed reruns.
- reward-component AUC analyses are diagnostic and should not be overstated as causal proof.

This section is critical because it answers the earlier concern that GRPO groups might have constant or noisy rewards.

## Qualitative Analysis

Use `outputs/dagig_paper_main_v1/paper_assets/case_studies/CASE_STUDY_SUMMARY.md`.

Main counts:

- KL-fixed seed42 dev: 7 method-only strict wins and 2 baseline-only losses.
- KL-fixed seed42 test: 5 method-only strict wins and 1 baseline-only loss.
- KL-fixed seed43 dev: 6 method-only strict wins and 1 baseline-only loss.
- KL-fixed seed43 test: 2 method-only strict wins and 0 baseline-only losses.

Use one or two wins and one loss:

- dev win: `pix2fact_11b37c2b51`, query adds home-loan/branch specificity and recovers answer `6`.
- test win: `pix2fact_e10ba14542`, query adds Auchan/Lego/French/date constraints and recovers `83`.
- loss: `pix2fact_9ac94cba26`, DAG-IG query loses the Ontario flag/province target and retrieves wrong phone evidence.

## Failure Analysis

Frame failures as remaining bottlenecks, not method invalidation:

- retrieval misses remain high: KL-fixed seed42 has 43 dev and 31 test retrieval misses; seed43 has 43 dev and 33 test retrieval misses.
- retrieved-evidence answer errors remain: both KL-fixed seeds have 10 dev and 7 test hit-answer-wrong cases.
- format failures and answer-in-query leakage are not the dominant issue.

This justifies future work on stronger query candidate generation and reader/verifier training, but not more same-recipe GRPO.

## Limitations

State explicitly:

- evaluation is offline BM25, not live web search.
- Pix2Fact sample size is limited.
- reader/answer extraction is still weak.
- gains are modest and paired tests are not individually significant.
- old-KL and goldfixed runs are diagnostic/control results, not the corrected headline.
- the method is node-level reward/credit, not a demonstrated causal counterfactual intervention.
- DAG-IG currently optimizes stage-1 policy only; full end-to-end retrieval/reader policy optimization remains future work.

## Figures And Tables Needed

1. Method diagram:
   image/question -> visual node -> query node -> evidence node -> answer node, with credit terms on each node.
2. Main results table:
   use `main_results_table.tex`.
3. Reward diagnostic table:
   use `node_credit_diagnostic_table.tex`.
4. Qualitative case table:
   compress selected rows from `CASE_STUDY_SUMMARY.md`.
5. Optional appendix table:
   negative/diagnostic runs showing why DAG-SFT, reranking, fusion, and answer repair are not the main path.

## What Not To Do Next

- Do not run another same-recipe GRPO.
- Do not make DAG-SFT the main method.
- Do not present DPO/pair scoring as complete.
- Do not claim answer repair or reader SFT solved the bottleneck.
- Do not use old-KL seed42 as the main checkpoint claim.
- Do not select by best test checkpoint.

## Efficient Next Step

Start the paper draft using this outline and the generated tables. If one additional artifact is needed before writing, make it a method diagram or a short appendix summary of discarded routes, not another training run.
