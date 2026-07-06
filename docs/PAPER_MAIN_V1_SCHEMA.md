# DAG-IG Paper Main v1 Schema

This is the only paper-main rollout schema for the next stage.

Pipeline:

```text
image + question
-> visual_observation
-> search_query
-> retrieve top-k evidence
-> final_answer
```

Each rollout row contains:

```json
{
  "sample_id": "...",
  "split": "train",
  "source_run": "dagig_grpo_full",
  "question": "...",
  "gold_answer": "...",
  "image_path": "...",
  "rollout": {
    "visual_observation": "...",
    "search_query": "...",
    "final_answer": "...",
    "raw": "...",
    "parsed_json": true
  },
  "retrieval": {
    "top_k": 5,
    "support_rank5": 1,
    "support_rank10": 1,
    "mrr10": 1.0,
    "hit5": true,
    "top_docs": []
  },
  "metrics": {
    "format_valid": true,
    "query_nonempty": true,
    "evidence_supported": true,
    "answer_correct": false,
    "strict_success": false,
    "answer_in_query": false
  },
  "node_credits": {
    "format_credit": 0.0,
    "visual_credit": 0.0,
    "query_credit": 0.0,
    "evidence_credit": 0.0,
    "answer_credit": 0.0,
    "leak_penalty": 0.0,
    "path_penalty": 0.0,
    "total_reward": 0.0
  }
}
```

Allowed reward-time information:

- generated visual_observation/search_query/final_answer;
- BM25 retrieval over the frozen train corpus during training;
- train support labels for train reward only;
- answer normalization against train gold answer for train reward only.

Disallowed:

- dev/test labels during training;
- teacher/oracle query as policy input;
- GPT/raw_pool/Qwen32B training data;
- URL/path-token shortcuts as positive query credit.
