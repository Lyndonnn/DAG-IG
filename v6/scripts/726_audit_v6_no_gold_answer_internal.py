#!/usr/bin/env python3
"""Private sealed-internal audit for clean v4 answer policies."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

METHODS=("no_credit","local_ig","outcome","dagig")
def rj(p):return json.loads(Path(p).read_text())
def rjl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for c in iter(lambda:f.read(1048576),b""):h.update(c)
 return h.hexdigest()
def load(path):
 spec=importlib.util.spec_from_file_location("dagig_answer_internal_eval_utils",path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument("--freeze",type=Path,required=True);p.add_argument("--train_fit",type=Path,required=True)
 for m in METHODS:p.add_argument(f"--{m}_scores",type=Path,required=True)
 p.add_argument("--output_dir",type=Path,required=True);a=p.parse_args();freeze=rj(a.freeze);fit=rj(a.train_fit)
 if fit.get("decision")!="DAGIG_V6_NO_GOLD_ANSWER_TRAIN_FIT_GO":raise ValueError("train fit is not GO")
 source_freeze=rj(rj(freeze["input_paths"]["terminal_audit"])["input_paths"]["freeze"])
 for key in ("private_labels","eval_utils"):
  if sha(source_freeze["input_paths"][key])!=source_freeze["input_hashes"][key]:raise ValueError(f"private audit input changed: {key}")
 labels={x["sample_id"]:x for x in rjl(source_freeze["input_paths"]["private_labels"])}
 evidence_audit=rj(freeze["input_paths"]["evidence_action_audit"])
 if sha(evidence_audit["output_paths"]["private_support"])!=evidence_audit["output_hashes"]["private_support"]:raise ValueError("private support audit changed")
 support={}
 for row in rjl(evidence_audit["output_paths"]["private_support"]):
  for strategy,value in row["strategy_support"].items():support[f"{row['query_id']}::{strategy}"]=bool(value)
 actions={x["answer_action_id"]:x for x in rjl(freeze["input_paths"]["answer_actions"])};targets=defaultdict(list)
 for row in rjl(freeze["input_paths"]["internal_data"]):targets[row["parent_group_id"]].append(row)
 helper=load(source_freeze["input_paths"]["eval_utils"]);metrics={};case_rows=[];inputs={"freeze":str(a.freeze.resolve()),"train_fit":str(a.train_fit.resolve())}
 for method in METHODS:
  audit_path=getattr(a,f"{method}_scores").resolve();audit=rj(audit_path)
  if audit.get("decision")!="DAGIG_V6_NO_GOLD_ANSWER_POLICY_SCORES_READY" or audit.get("method")!=method or audit.get("partition")!="internal_holdout":raise ValueError(f"invalid internal scores: {method}")
  for key,path in audit["output_paths"].items():
   if sha(path)!=audit["output_hashes"][key]:raise ValueError(f"score output changed: {method}/{key}")
  scores=rjl(audit["output_paths"]["scores"]);generations={x["parent_group_id"]:x for x in rjl(audit["output_paths"]["generations"])};selected=[];generated=[]
  for score in scores:
   group=sorted(targets[score["parent_group_id"]],key=lambda x:x["answer_action_id"]);idx=max(range(len(group)),key=score["policy_probabilities"].__getitem__);row=group[idx];answer=actions[row["answer_action_id"]]["candidate_answer"];label=labels[row["sample_id"]];match=helper.answer_match_details(answer,label["gold_answer"],label.get("aliases") or []);sup=support[row["parent_group_id"]]
   selected.append({"answer_correct":bool(match["answer_correct"]),"support":sup,"strict":bool(sup and match["answer_correct"]),"expected_terminal":sum(float(x["child_success_probability"])*p for x,p in zip(group,score["policy_probabilities"]))})
   gen=generations[row["parent_group_id"]];gm=helper.answer_match_details(gen["final_answer"],label["gold_answer"],label.get("aliases") or []);generated.append({"valid":bool(gen["valid_json"]),"answer_correct":bool(gm["answer_correct"]),"support":sup,"strict":bool(sup and gm["answer_correct"])})
   case_rows.append({"method":method,"sample_id":row["sample_id"],"parent_group_id":row["parent_group_id"],"selected_action_id":row["answer_action_id"],"candidate_answer_correct":selected[-1]["answer_correct"],"evidence_supported":sup,"candidate_strict":selected[-1]["strict"],"generated_valid":generated[-1]["valid"],"generated_answer_correct":generated[-1]["answer_correct"],"generated_strict":generated[-1]["strict"]})
  supported=[x for x in generated if x["support"]];metrics[method]={"groups":len(selected),"candidate_answer_correct":mean(x["answer_correct"] for x in selected),"candidate_strict":mean(x["strict"] for x in selected),"model_expected_terminal":mean(x["expected_terminal"] for x in selected),"generated_valid_json":mean(x["valid"] for x in generated),"generated_answer_correct":mean(x["answer_correct"] for x in generated),"generated_strict":mean(x["strict"] for x in generated),"generated_answer_correct_given_support":mean(x["answer_correct"] for x in supported) if supported else 0.0}
  inputs[f"{method}_score_audit"]=str(audit_path)
 strongest=max(metrics[m]["generated_strict"] for m in ("local_ig","outcome"));gates={"complete_equal_internal_groups":set(x["groups"] for x in metrics.values())=={480},"all_generation_valid_at_least_0p98":all(x["generated_valid_json"]>=.98 for x in metrics.values()),"dagig_candidate_strict_not_below_no_credit":metrics["dagig"]["candidate_strict"]>=metrics["no_credit"]["candidate_strict"],"dagig_generated_strict_not_below_no_credit":metrics["dagig"]["generated_strict"]>=metrics["no_credit"]["generated_strict"],"dagig_generated_noninferior_to_strong_control":metrics["dagig"]["generated_strict"]+.01>=strongest,"dagig_expected_terminal_not_below_no_credit":metrics["dagig"]["model_expected_terminal"]>=metrics["no_credit"]["model_expected_terminal"],"internal_holdout_unused_for_training":True,"dev_sealed":True,"test_sealed":True}
 decision="DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_GO" if all(gates.values()) else "DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_NO_GO";out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);cases=out/"v6_no_gold_answer_internal_private_cases.jsonl";cases.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in case_rows))
 result={"decision":decision,"metrics":metrics,"gates":gates,"input_paths":inputs,"input_hashes":{k:sha(v) for k,v in inputs.items()},"output_paths":{"private_cases":str(cases)},"output_hashes":{"private_cases":sha(cases)},"gold_or_qrels_loaded_only_by_private_auditor":True,"internal_holdout_used_for_training":False,"dev_used":False,"test_used":False,"training_run":False};path=out/"DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_AUDIT.json";path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
