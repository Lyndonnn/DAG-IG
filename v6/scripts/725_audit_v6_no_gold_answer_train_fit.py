#!/usr/bin/env python3
"""Audit matched clean v4 answer-policy fit before opening the internal holdout."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
from statistics import mean

METHODS=("no_credit","local_ig","outcome","dagig")
def rj(p): return json.loads(Path(p).read_text())
def rjl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for c in iter(lambda:f.read(1048576),b""): h.update(c)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument("--freeze",type=Path,required=True)
 for m in METHODS:p.add_argument(f"--{m}_scores",type=Path,required=True)
 p.add_argument("--output_dir",type=Path,required=True); a=p.parse_args(); freeze=rj(a.freeze); targets=defaultdict(list)
 for row in rjl(freeze["input_paths"]["train_data"]): targets[row["parent_group_id"]].append(row)
 metrics={}; inputs={"freeze":str(a.freeze.resolve())}
 for method in METHODS:
  audit_path=getattr(a,f"{method}_scores").resolve(); audit=rj(audit_path)
  if audit.get("decision")!="DAGIG_V6_NO_GOLD_ANSWER_POLICY_SCORES_READY" or audit.get("method")!=method or audit.get("partition")!="policy_train": raise ValueError(f"invalid train scores: {method}")
  rows=rjl(audit["output_paths"]["scores"]); tvs=[]; top=[]; high=[]
  key=freeze["target_keys"][method]
  for score in rows:
   group=sorted(targets[score["parent_group_id"]],key=lambda x:x["answer_action_id"]); target=[float(x[key]) for x in group]; model=score["policy_probabilities"]
   tvs.append(.5*sum(abs(x-y) for x,y in zip(target,model))); ti=max(range(len(target)),key=target.__getitem__); mi=max(range(len(model)),key=model.__getitem__); top.append(ti==mi)
   ordered=sorted(target,reverse=True); margin=ordered[0]-ordered[1]
   if margin>=.10: high.append(ti==mi)
  metrics[method]={"groups":len(rows),"mean_tv":mean(tvs),"top_action_agreement":mean(top),"high_margin_groups":len(high),"high_margin_agreement":mean(high) if high else 1.0}
  inputs[f"{method}_score_audit"]=str(audit_path)
 gates={"complete_equal_groups":set(x["groups"] for x in metrics.values())=={int(freeze["parent_groups"])},"all_mean_tv_at_most_0p10":all(x["mean_tv"]<=.10 for x in metrics.values()),"all_top_agreement_at_least_0p70":all(x["top_action_agreement"]>=.70 for x in metrics.values()),"all_high_margin_agreement_at_least_0p90":all(x["high_margin_agreement"]>=.90 for x in metrics.values()),"internal_holdout_still_sealed":True,"dev_sealed":True,"test_sealed":True}
 decision="DAGIG_V6_NO_GOLD_ANSWER_TRAIN_FIT_GO" if all(gates.values()) else "DAGIG_V6_NO_GOLD_ANSWER_TRAIN_FIT_NO_GO"; out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 result={"decision":decision,"metrics":metrics,"gates":gates,"input_paths":inputs,"input_hashes":{k:sha(v) for k,v in inputs.items()},"gold_or_qrels_loaded":False,"internal_holdout_used":False,"dev_used":False,"test_used":False,"training_run":False}; path=out/"DAGIG_V6_NO_GOLD_ANSWER_TRAIN_FIT_AUDIT.json";path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
