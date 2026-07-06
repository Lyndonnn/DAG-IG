# DAG-IG: Node-Level Credit Assignment for Long-Horizon Multimodal Search Agents

> Status: superseded draft. Use `paper/latex/main.tex`, `results/reports/KLFIXED_GRPO_60_REPORT.md`, and `results/reports/CORPUS_REALITY_AUDIT.md` for the corrected paper-facing claims. This file is retained as a historical prose draft and must not be used as the authoritative result source.

## Abstract

Multimodal search agents must convert visual observations into search queries, retrieve evidence, and extract an answer. A final-answer reward alone is too sparse to identify which part of this chain caused success or failure. We introduce DAG-IG, a node-level credit assignment method for long-horizon multimodal search agents. DAG-IG represents each rollout as a directed computation graph over a visual node, query node, evidence node, and answer node, then assigns reward to each node according to grounded retrieval and final answer success. In a frozen Pix2Fact evidence-note BM25 setting, KL-fixed DAG-IG GRPO improves strict success over a Format-SFT two-stage baseline from 40.8% to 45.9% on dev and from 34.4% to 39.1% on test as a two-seed mean. A fixed-reader control obtains the same strict result, showing the gain is not explained by reader drift. The gains are modest and not individually significant on the small test set, but they are directionally consistent under a corrected KL penalty and stricter answer checker.

## 1. Introduction

Many visually grounded questions cannot be answered from the image alone. A model may need to identify an entity in the image, formulate a search query, retrieve external evidence, and then extract a precise answer. This creates a long-horizon multimodal search problem: early visual and query decisions determine which evidence is available, while the final answer is only observed after several intermediate decisions.

Training such agents with final-answer reward is inefficient. If an answer is wrong, the failure may come from an incorrect visual observation, an underspecified query, a missing evidence document, or a reader that ignored the correct evidence. Treating all of these failures as a single scalar outcome makes credit assignment noisy and often pushes models toward unstable query or answer behavior.

We study this problem on Pix2Fact, where questions require image understanding plus external fact retrieval. Our agent follows a fixed two-stage rollout:

```text
image + question
-> visual_observation
-> search_query
-> retrieve top-k evidence
-> final_answer
```

The central idea of DAG-IG is to assign credit at the nodes of this rollout graph. Rather than rewarding only the final answer, DAG-IG separately scores the visual observation, query, retrieved evidence, and answer. These node-level credits are combined into a reward for grouped GRPO, optimizing the policy that emits the visual observation and search query.

This paper makes five contributions:

1. We formulate multimodal search rollouts as a directed graph with visual, query, evidence, and answer nodes.
2. We propose DAG-IG node-level credit for separating retrieval, evidence, and answer contributions.
3. We implement a two-stage GRPO training loop that optimizes the stage-1 visual/query policy while keeping retrieval and reader evaluation fixed and auditable.
4. We show that DAG-IG improves a Format-SFT two-stage agent in a frozen offline Pix2Fact retrieval setting.
5. We audit the reward itself, showing that it is non-collapsed and predictive of retrieval and strict success.

The result should be interpreted carefully. DAG-IG is not yet a solved end-to-end web-search agent. The current evidence is an offline BM25 evaluation with a limited Pix2Fact split, and the reader remains a bottleneck. The contribution is a clean, auditable training signal for long-horizon multimodal search, not a claim that answer extraction or live web search is solved.

## 2. Task And Agent Setup

Each example consists of an image, a natural-language question, a gold answer, and an offline evidence corpus. The model must answer the question using information grounded in the image and supported by retrieved evidence.

We evaluate a two-stage agent. In stage 1, the multimodal policy receives the image and question and emits compact JSON:

```json
{
  "visual_observation": "...",
  "search_query": "..."
}
```

The search query is passed to a frozen BM25 retriever over the offline corpus, returning top-5 documents. In stage 2, a reader prompt receives the image, question, and retrieved evidence and emits:

```json
{
  "final_answer": "..."
}
```

Evaluation reports retrieval hit at rank 1, 3, and 5; answer correctness; evidence support; and strict success. Strict success requires both answer correctness and evidence support. We also track format parse success and answer-in-query leakage.

This separation is important: the policy being optimized controls the visual/query stage, while the retriever and reader setting remain fixed during evaluation. This makes it possible to distinguish improvements in search behavior from reader drift.

## 3. DAG-IG Method

### 3.1 DAG-Structured Rollout

Let the input be \(x=(I,q)\), where \(I\) is the image and \(q\) is the question. The stage-1 policy emits a visual observation \(z_v\) and search query \(z_q\):

\[
(z_v,z_q) \sim \pi_\theta(\cdot \mid I,q).
\]

The retriever returns evidence:

\[
z_e = \mathrm{BM25}(z_q, k).
\]

The reader produces the final answer \(y\) conditioned on the image, question, and evidence. This gives a directed rollout graph:

\[
I,q \rightarrow z_v \rightarrow z_q \rightarrow z_e \rightarrow y.
\]

### 3.2 Node-Level Credit

DAG-IG assigns credit to each node:

- \(C_v\): visual credit for preserving useful visual anchors or entities.
- \(C_q\): query credit for retrieving support evidence.
- \(C_e\): evidence credit for whether retrieved documents contain support.
- \(C_a\): answer credit for evidence-supported answer correctness.
- \(C_f\): format credit for valid compact JSON and required fields.
- \(P_{\mathrm{leak}}\), \(P_{\mathrm{path}}\): penalties for answer-in-query leakage and degenerate paths.

The rollout reward is:

\[
R(x,z_v,z_q,z_e,y)
= w_v C_v + w_q C_q + w_e C_e + w_a C_a + w_f C_f
- P_{\mathrm{leak}} - P_{\mathrm{path}}.
\]

The exact component implementation is deliberately auditable: query and evidence terms are tied to retrieval/support outcomes, answer credit is tied to final answer correctness under evidence support, and format credit is low-variance so it does not dominate ranking.

### 3.3 Grouped GRPO Optimization

For each training sample, we sample a group of candidate rollouts. We compute DAG-IG reward for each rollout and derive group-relative advantages. GRPO then optimizes the stage-1 policy:

\[
\max_\theta
\mathbb{E}
\left[
A_i \log \pi_\theta(z_{v,i},z_{q,i}\mid x)
- \beta \mathrm{KL}(\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{init}}(\cdot\mid x))
\right].
\]

Only the stage-1 visual/query policy receives policy-gradient updates. Retrieval is deterministic, and the reader prompt is fixed for evaluation. This design prevents the model from hiding query failures behind a changing reader.

## 4. Experimental Setup

### 4.1 Data And Retrieval

We use the clean Pix2Fact diagnostic setup with an offline BM25 corpus. Dev and test evaluation use a frozen corpus and fixed top-5 retrieval. We do not use real web search. Teacher/oracle queries are not part of dev/test evaluation. Old unclean data and oracle trajectories are excluded from the paper-main training/evaluation path.

### 4.2 Models

The base model is Qwen2.5-VL-3B-Instruct. The main initializer is a Format-SFT adapter that teaches the two-stage JSON format. DAG-IG GRPO starts from this initializer and optimizes the stage-1 policy.

The main training recipe uses:

- two-stage rollout;
- stage1-only policy loss;
- top-k retrieval \(k=5\);
- grouped GRPO with 4 generations per sample;
- KL coefficient 0.1 with a non-negative k3 estimator;
- learning rate \(1\times10^{-6}\);
- 60 optimizer steps for the main run.

### 4.3 Baselines And Controls

The paper-main comparison includes:

- Format-SFT v4: the stage-1 format baseline under checker v4.
- KL-fixed DAG-IG seed42: corrected k3-KL rerun.
- KL-fixed DAG-IG seed43: corrected k3-KL second seed.
- KL-fixed two-seed mean: the corrected headline result.

Other routes, including verbose DAG-SFT, outcome-only SFT, preference/DPO pilots, query reranking, multi-query fusion, and broad answer repair, are diagnostics. They motivated the current method but are not the main result.

## 5. Main Results

The main evaluation is shown below.

| Method | Split | R@1 | R@3 | R@5 | Ans. | Strict |
|---|---|---:|---:|---:|---:|---:|
| Format-SFT v4 | dev | 35.7 | 49.0 | 52.0 | 42.9 | 40.8 |
| Format-SFT v4 | test | 31.2 | 43.8 | 46.9 | 34.4 | 34.4 |
| KL-fixed DAG-IG seed42 | dev | 36.7 | 51.0 | 56.1 | 48.0 | 45.9 |
| KL-fixed DAG-IG seed42 | test | 39.1 | 46.9 | 51.6 | 40.6 | 40.6 |
| KL-fixed DAG-IG seed43 | dev | 36.7 | 51.0 | 56.1 | 48.0 | 45.9 |
| KL-fixed DAG-IG seed43 | test | 34.4 | 43.8 | 48.4 | 37.5 | 37.5 |
| KL-fixed two-seed mean | dev | 36.7 | 51.0 | 56.1 | 48.0 | 45.9 |
| KL-fixed two-seed mean | test | 36.8 | 45.4 | 50.0 | 39.1 | 39.1 |

KL-fixed DAG-IG improves strict success over Format-SFT by +5.1 points on dev and +4.7 points on test as a two-seed mean. Retrieval R@5 improves from 52.0% to 56.1% on dev and from 46.9% to 50.0% on test. Seed42 individually reaches 40.6% test strict, while seed43 reaches 37.5%; the clean headline is therefore the two-seed mean rather than the best test seed.

## 6. Reward And Credit Diagnostics

Because GRPO can fail if reward groups are constant or noisy, we audit reward quality directly.

| Run | Optimizer steps | Micro steps | Constant groups | Constant rate |
|---|---:|---:|---:|---:|
| KL-fixed smoke v2 | 1 | 4 | 0 | 0.0% |
| KL-fixed seed42 | 60 | 240 | 3 | 1.2% |
| KL-fixed seed43 | 60 | 240 | 1 | 0.4% |

The KL-fixed reruns have low constant-reward rates, so the earlier high-constant-reward concern does not apply. Earlier reward-component AUC analyses remain useful diagnostics, but they should not be interpreted as causal counterfactual intervention evidence.

This audit is central to the paper claim. The method is not just a successful checkpoint; the node-level credit signal itself is measurable, though the current evidence should be described as diagnostic rather than causal.

## 7. Qualitative Analysis

Compared with Format-SFT, KL-fixed seed42 has 7 dev strict-only wins and 2 dev strict-only losses. On test, it has 5 strict-only wins and 1 strict-only loss. KL-fixed seed43 has 6 dev strict-only wins and 1 dev strict-only loss, and 2 test strict-only wins with 0 test strict-only losses.

One dev win is `pix2fact_11b37c2b51`. Format-SFT queries `Bank of America Los Angeles branches` and answers `3`, missing support. DAG-IG queries `Bank of America branches in Los Angeles offering home loan services`, retrieves support, and answers `6`.

Another dev win is `pix2fact_522951e822`. Format-SFT queries `bank with red logo in Hong Kong` and answers `HSBC`. DAG-IG queries `HSBC bank in Hong Kong`, retrieves support documents, and answers the requested BIC `TUBDDEDDXXX`.

On test, `pix2fact_e10ba14542` illustrates a retrieval gain. Format-SFT queries only `Auchan欧尚` and fails. DAG-IG adds the shopping target and time context with `Auchan欧尚 Lego Duplo products in France April 2026`, retrieves support, and answers `83`.

DAG-IG also has losses. In `pix2fact_9ac94cba26`, Format-SFT retrieves the correct Ontario contact answer, while DAG-IG drifts to a generic query about flags in a Canadian building and answers a wrong phone number. These losses show that query specificity and visual disambiguation remain open problems.

## 8. Failure Analysis

The main remaining failures are not output format or answer leakage. Format parse success is near-perfect, and answer-in-query leakage is zero or near zero in the main runs.

For KL-fixed seed42, dev/test retrieval misses are 43 and 31. Retrieved-evidence answer-wrong cases are 10 and 7. This indicates two bottlenecks:

1. The stage-1 policy still sometimes fails to formulate a query that retrieves support.
2. Even when evidence is retrieved, the reader sometimes extracts the wrong span or answers at the wrong granularity.

Several attempted fixes did not become the main path. Broad answer repair created false repairs and did not improve test. Lightweight answer verifiers and small reader-SFT runs did not solve the reader bottleneck. Query warmup improved retrieval in isolation but did not beat the main GRPO strict result. These negative results support the current design decision: keep the main claim focused on node-level credit and avoid claiming that retrieval or reading is solved.

## 9. Limitations

The experiment is offline and uses a frozen Pix2Fact evidence-note BM25 corpus, so it does not establish live web-search generalization or robustness to full noisy web pages. The split is also modest, and paired tests are not individually significant, so effect sizes should be interpreted as diagnostic but meaningful rather than definitive large-scale performance. The reader remains weak, and answer extraction errors limit strict success even when retrieval succeeds.

DAG-IG currently optimizes the visual/query stage only. Full end-to-end optimization of visual grounding, retrieval, evidence selection, and answer extraction remains future work. Old-KL and goldfixed runs are retained as diagnostics/controls, not as the corrected headline result.

## 10. Conclusion

DAG-IG addresses a central problem in long-horizon multimodal search: final-answer reward is too sparse to tell which intermediate decision caused success or failure. By assigning node-level credit to visual, query, evidence, and answer stages, DAG-IG provides a discriminative reward for grouped GRPO. In the Pix2Fact evidence-note BM25 setting, the KL-fixed two-seed mean improves a Format-SFT two-stage agent on both dev and test. Reward audits show the training signal is non-collapsed. The current system is not a complete web-search agent, but it establishes a practical, auditable route for training multimodal agents with structured credit rather than final-answer-only supervision.

## Appendix Pointers

- Main table: `outputs/dagig_paper_main_v1/paper_assets/main_results_table.tex`
- Node-credit table: `outputs/dagig_paper_main_v1/paper_assets/node_credit_diagnostic_table.tex`
- Method diagram: `outputs/dagig_paper_main_v1/paper_assets/figures/dagig_method_diagram.tex`
- Reward equations: `outputs/dagig_paper_main_v1/paper_assets/figures/dagig_reward_equations.tex`
- Case studies: `outputs/dagig_paper_main_v1/paper_assets/case_studies/CASE_STUDY_SUMMARY.md`
- Corrected evidence brief: `results/reports/PAPER_MAIN_EVIDENCE_BRIEF.md`
- KL-fixed report: `results/reports/KLFIXED_GRPO_60_REPORT.md`
- Corpus reality audit: `results/reports/CORPUS_REALITY_AUDIT.md`
