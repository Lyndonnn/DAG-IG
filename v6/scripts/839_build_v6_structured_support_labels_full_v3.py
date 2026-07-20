#!/usr/bin/env python3
"""Aggregate structured support v3 shards into action-level private labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--shard_dirs",type=Path,nargs="+",required=True)
    parser.add_argument("--output_dir",type=Path,required=True)
    parser.add_argument("--amended_from_freeze",type=Path)
    args=parser.parse_args()
    freeze_path=args.freeze.resolve(); freeze=read_json(freeze_path)
    if freeze.get("decision")!="DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_FROZEN": raise ValueError("Full v3 protocol not frozen")
    if freeze["input_hashes"]["builder"]!=sha256(Path(__file__).resolve()): raise ValueError("Builder changed after freeze")
    old_freeze=None; amendment=None
    if args.amended_from_freeze:
        old_path=args.amended_from_freeze.resolve();old_freeze=read_json(old_path)
        invariants={
            "same_prompts":old_freeze["output_hashes"]["remaining_prompts"]==freeze["output_hashes"]["remaining_prompts"],
            "same_seed_decisions":old_freeze["output_hashes"]["pilot_seed_decisions"]==freeze["output_hashes"]["pilot_seed_decisions"],
            "same_state_map":old_freeze["output_hashes"]["state_map"]==freeze["output_hashes"]["state_map"],
            "same_model":old_freeze["model"]==freeze["model"],"same_generation":old_freeze["generation"]==freeze["generation"],"same_sharding":old_freeze["sharding"]==freeze["sharding"],
        }
        if not all(invariants.values()):raise ValueError(f"Unsafe builder amendment: {invariants}")
        amendment={"old_freeze":str(old_path),"old_freeze_sha256":sha256(old_path),"new_freeze":str(freeze_path),"new_freeze_sha256":sha256(freeze_path),"invariants":invariants}
    decisions={}
    manifests=[]
    for directory in args.shard_dirs:
        manifest=read_json(directory.resolve()/"SHARD_MANIFEST.json")
        allowed_hashes={sha256(freeze_path)}
        if old_freeze is not None:allowed_hashes.add(amendment["old_freeze_sha256"])
        if manifest.get("decision")!="DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_SHARD_COMPLETE" or manifest["freeze_sha256"] not in allowed_hashes: raise ValueError(f"Invalid shard {directory}")
        path=Path(manifest["score_path"])
        if sha256(path)!=manifest["score_sha256"]: raise ValueError(f"Changed shard {directory}")
        manifests.append(manifest)
        for row in read_jsonl(path):
            if row["support_state_id"] in decisions: raise ValueError("Duplicate scored state")
            decisions[row["support_state_id"]]=row
    if sorted(row["shard_index"] for row in manifests)!=list(range(int(freeze["sharding"]["num_shards"]))): raise ValueError("Shard set incomplete")
    seed_path=Path(freeze["output_paths"]["pilot_seed_decisions"])
    if sha256(seed_path)!=freeze["output_hashes"]["pilot_seed_decisions"]: raise ValueError("Pilot seeds changed")
    for row in read_jsonl(seed_path):
        state_id=row["support_state_id"]
        if state_id in decisions: raise ValueError("Seed/scored overlap")
        decisions[state_id]=row
    state_map_path=Path(freeze["output_paths"]["state_map"])
    if sha256(state_map_path)!=freeze["output_hashes"]["state_map"]: raise ValueError("State map changed")
    states={row["support_state_id"]:row for row in read_jsonl(state_map_path)}
    if set(states)!=set(decisions): raise ValueError(f"Full state coverage mismatch {len(states)}/{len(decisions)}")
    action_rows=[]
    for state_id,state in sorted(states.items()):
        decision=decisions[state_id]
        for action_id in state["evidence_action_ids"]:
            action_rows.append({
                "evidence_action_id":action_id,
                "support_state_id":state_id,
                "sample_id":state["sample_id"],
                "partition":state["partition"],
                "structured_support_label":bool(decision["supported"]),
                "supporting_doc_indices":decision["supporting_doc_indices"],
                "supporting_span_private":decision["supporting_span"],
                "entailment_type":decision["entailment_type"],
                "conflict_present":bool(decision["conflict_present"]),
                "teacher_model":decision["model"],
                "label_status":"structured_v3_contract_pilot_audited_go",
            })
    if len(action_rows)!=14770 or len({row["evidence_action_id"] for row in action_rows})!=14770: raise ValueError("Action label universe mismatch")
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    labels_path=output/"v6_structured_support_v3_action_labels_private.jsonl"
    with labels_path.open("w",encoding="utf-8") as handle:
        for row in sorted(action_rows,key=lambda value:value["evidence_action_id"]): handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    train_labels_path=output/"v6_structured_support_v3_policy_train_labels_private.jsonl"
    internal_labels_path=output/"v6_structured_support_v3_internal_holdout_labels_private.jsonl"
    with train_labels_path.open("w",encoding="utf-8") as train_handle, internal_labels_path.open("w",encoding="utf-8") as internal_handle:
        for row in sorted(action_rows,key=lambda value:value["evidence_action_id"]):
            target=train_handle if row["partition"]=="policy_train" else internal_handle
            target.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    counts=Counter((row["partition"],row["structured_support_label"]) for row in action_rows)
    summary={
        "decision":"DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_GO_FOR_SELECTOR_DEVELOPMENT",
        "actions":len(action_rows),"states":len(states),"pilot_seed_states":int(freeze["counts"]["pilot_seed_states"]),
        "label_counts":{f"{part}::{label}":count for (part,label),count in sorted(counts.items())},
        "positive_rate":{part:sum(row["structured_support_label"] for row in action_rows if row["partition"]==part)/sum(row["partition"]==part for row in action_rows) for part in ("policy_train","internal_holdout")},
        "labels_path":str(labels_path),"labels_sha256":sha256(labels_path),
        "policy_train_labels_path":str(train_labels_path),"policy_train_labels_sha256":sha256(train_labels_path),
        "internal_holdout_labels_path":str(internal_labels_path),"internal_holdout_labels_sha256":sha256(internal_labels_path),
        "runtime_policy_may_read_private_labels":False,
        "selector_value_supervision_allowed":True,
        "final_paper_claim_requires_additional_human_spot_audit":True,
        "protocol_amendment":amendment,
        "serper_calls":0,"dev_used":False,"test_used":False,"training_run":False,
    }
    summary_path=output/"DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_SUMMARY.json"
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
