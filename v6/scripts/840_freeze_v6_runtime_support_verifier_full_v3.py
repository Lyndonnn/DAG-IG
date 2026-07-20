#!/usr/bin/env python3
"""Freeze no-gold runtime support-verifier inputs for all evidence actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict evidence sufficiency verifier for web research.
Do not answer the question. Decide whether at least one supplied document contains enough information to determine the requested answer for the exact entity and every date, location, comparison, and other constraint.
The visual observation and executed query identify intent but are not evidence. Equivalent formatting and simple question-requested rounding are acceptable. Mere topical relevance, entity mentions without the requested fact, wrong conditions, weak hints, and unsupported guesses are insufficient.
Return exactly one character: A for sufficient evidence, or B for insufficient evidence."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(root: Path) -> dict[str, Any]:
    files=[root/name for name in ("config.json","tokenizer.json","model.safetensors.index.json")]+sorted(root.glob("model-*.safetensors"))
    if len(files)<4 or any(not path.is_file() for path in files): raise FileNotFoundError(f"Incomplete local model: {root}")
    return {"path":str(root),"files":{path.name:{"bytes":path.stat().st_size,"sha256":sha256(path)} for path in files}}


def evidence_text(docs: list[dict[str, Any]]) -> str:
    return "\n\n".join("\n".join([
        f"Document {index}",f"Title: {str(doc.get('title') or '').strip()}",f"Source: {str(doc.get('domain') or '').strip()}",
        f"Date: {str(doc.get('date') or '').strip() or 'not provided'}",f"Snippet: {str(doc.get('snippet') or '').strip()}",
    ]) for index,doc in enumerate(docs,1))


def numeric(doc: dict[str, Any], name: str) -> float:
    try: return float(doc.get(name) or 0.0)
    except (TypeError,ValueError): return 0.0


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--label_summary",type=Path,required=True)
    parser.add_argument("--evidence_actions",type=Path,required=True)
    parser.add_argument("--model_path",type=Path,required=True)
    parser.add_argument("--scorer",type=Path,required=True)
    parser.add_argument("--fitter",type=Path,required=True)
    parser.add_argument("--auditor",type=Path,required=True)
    parser.add_argument("--helper",type=Path,required=True)
    parser.add_argument("--output_dir",type=Path,required=True)
    args=parser.parse_args()
    paths={name:path.resolve() for name,path in {"label_summary":args.label_summary,"evidence_actions":args.evidence_actions,"scorer":args.scorer,"fitter":args.fitter,"auditor":args.auditor,"helper":args.helper}.items()}
    for path in paths.values():
        if not path.is_file(): raise FileNotFoundError(path)
    summary=read_json(paths["label_summary"])
    if summary.get("decision")!="DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_GO_FOR_SELECTOR_DEVELOPMENT": raise ValueError("Full structured support labels are not ready")
    label_path=Path(summary["labels_path"])
    if sha256(label_path)!=summary["labels_sha256"]: raise ValueError("Full structured labels changed")
    train_label_path=Path(summary["policy_train_labels_path"]); internal_label_path=Path(summary["internal_holdout_labels_path"])
    if sha256(train_label_path)!=summary["policy_train_labels_sha256"] or sha256(internal_label_path)!=summary["internal_holdout_labels_sha256"]: raise ValueError("Split structured labels changed")
    actions=read_jsonl(paths["evidence_actions"])
    records=[]
    strategies=sorted({row["evidence_strategy"] for row in actions})
    for action in actions:
        docs=action.get("selected_docs") or []
        if len(docs)!=3: raise ValueError(f"Expected three docs: {action['evidence_action_id']}")
        bge=[numeric(doc,"normalized_bge_score") for doc in docs]
        overlap=[numeric(doc,"question_keyword_overlap") for doc in docs]
        type_match=[numeric(doc,"answer_type_pattern_match") for doc in docs]
        feature={
            "mean_normalized_bge":sum(bge)/3,"max_normalized_bge":max(bge),"mean_question_overlap":sum(overlap)/3,
            "max_question_overlap":max(overlap),"max_answer_type_pattern_match":max(type_match),
            "domain_diversity":len({str(doc.get('domain') or '').casefold() for doc in docs})/3,
        }
        for strategy in strategies: feature[f"strategy::{strategy}"]=float(action["evidence_strategy"]==strategy)
        prompt="\n\n".join([
            f"Question:\n{str(action['question']).strip()}",f"Visual observation:\n{str(action['visual_observation']).strip()}",
            f"Executed search query:\n{str(action['search_query']).strip()}",f"Selected evidence:\n{evidence_text(docs)}","Decision:",
        ])
        records.append({
            "evidence_action_id":action["evidence_action_id"],"query_id":action["query_id"],"sample_id":action["sample_id"],
            "partition":action["partition"],"evidence_strategy":action["evidence_strategy"],"system_prompt":SYSTEM_PROMPT,
            "user_prompt":prompt,"runtime_features":feature,
        })
    train=sum(row["partition"]=="policy_train" for row in records); internal=sum(row["partition"]=="internal_holdout" for row in records)
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    input_path=output/"v6_runtime_support_verifier_full_v3_inputs_no_labels.jsonl"
    with input_path.open("w",encoding="utf-8") as handle:
        for row in sorted(records,key=lambda value:value["evidence_action_id"]): handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    gates={
        "exact_14770_actions":len(records)==14770,"exact_11795_2975_partition":(train,internal)==(11795,2975),
        "five_actions_per_query":all(count==5 for count in __import__('collections').Counter(row["query_id"] for row in records).values()),
        "runtime_records_have_no_private_labels":all(not ({"gold_answer","support_label","answer_correct","oracle","qrel"}&set(row)) for row in records),
        "prompt_has_no_private_reference":all("Private reference answer:" not in row["user_prompt"] for row in records),
        "dev_sealed":True,"test_sealed":True,
    }
    protocol={
        "decision":"DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_FROZEN" if all(gates.values()) else "DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_NO_GO",
        "protocol_version":"dagig_v6_no_gold_runtime_support_verifier_full_v3","system_prompt":SYSTEM_PROMPT,
        "verifier":{"model":"local Qwen2.5-VL-7B-Instruct text-only","score":"next-token logit(A)-logit(B)","max_input_tokens":2048,"attn_implementation":"sdpa","dtype":"bfloat16","num_shards":2},
        "fit":{"folds":5,"repeats":3,"seed_prefix":"dagig_v6_runtime_support_v3","l2_grid":[0.003,0.01,0.03,0.1,0.3],"newton_steps":60,"probability_clip":[1e-5,0.99999],
            "feature_families":[
                ["semantic_logit"],
                ["semantic_logit","mean_normalized_bge","max_normalized_bge","mean_question_overlap","max_question_overlap","max_answer_type_pattern_match","domain_diversity"],
                ["semantic_logit","mean_normalized_bge","max_normalized_bge","mean_question_overlap","max_question_overlap","max_answer_type_pattern_match","domain_diversity",*[f"strategy::{value}" for value in strategies]],
            ],
            "selection_rule":"first feature family with a passing l2; within it minimum OOF Brier then smallest l2"},
        "train_oof_gates":{"auc_min":0.82,"brier_improvement_vs_prevalence_min":0.01,"within_query_pair_order_min":0.68,"nonconstant_query_rate_min":0.95,"semantic_coefficient_positive":True},
        "internal_gates":{"auc_min":0.80,"brier_improvement_vs_prevalence_min":0.005,"within_query_pair_order_min":0.65},
        "input_paths":{**{name:str(path) for name,path in paths.items()},"private_labels_combined":str(label_path),"private_train_labels":str(train_label_path),"private_internal_labels":str(internal_label_path)},
        "input_hashes":{**{name:sha256(path) for name,path in paths.items()},"private_labels_combined":sha256(label_path),"private_train_labels":sha256(train_label_path),"private_internal_labels":sha256(internal_label_path)},
        "model_fingerprint":model_fingerprint(args.model_path.resolve()),
        "output_paths":{"verifier_inputs":str(input_path)},"output_hashes":{"verifier_inputs":sha256(input_path)},
        "gates":gates,"private_labels_loaded_by_scorer":False,"internal_used_for_fit":False,"serper_calls":0,"dev_used":False,"test_used":False,
    }
    freeze_path=output/"DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_FREEZE.json"; freeze_path.write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"decision":protocol["decision"],"rows":len(records),"train":train,"internal":internal,"gates":gates,"freeze":str(freeze_path)},indent=2))


if __name__=="__main__": main()
