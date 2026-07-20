#!/usr/bin/env python3
"""Freeze corrected no-gold answer-action terminal values after support GO."""

from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
from typing import Any
def rj(p:Path)->dict[str,Any]:return json.loads(p.read_text())
def rjl(p:Path)->list[dict[str,Any]]:
 with p.open() as h:return [json.loads(x) for x in h if x.strip()]
def sh(p:Path)->str:
 d=hashlib.sha256()
 with p.open("rb") as h:
  for c in iter(lambda:h.read(1024*1024),b""):d.update(c)
 return d.hexdigest()
def logit(v:float)->float:
 v=min(max(v,1e-5),1-1e-5);return math.log(v/(1-v))
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--support_freeze",type=Path,required=True);p.add_argument("--support_train_audit",type=Path,required=True);p.add_argument("--support_internal_audit",type=Path,required=True);p.add_argument("--terminal_scores",type=Path,nargs="+",required=True);p.add_argument("--terminal_private",type=Path,required=True);p.add_argument("--shared_answer_values",type=Path,required=True);p.add_argument("--fitter",type=Path,required=True);p.add_argument("--auditor",type=Path,required=True);p.add_argument("--helper",type=Path,required=True);p.add_argument("--output_dir",type=Path,required=True);a=p.parse_args()
 paths={"support_freeze":a.support_freeze.resolve(),"support_train_audit":a.support_train_audit.resolve(),"support_internal_audit":a.support_internal_audit.resolve(),"terminal_private":a.terminal_private.resolve(),"shared_answer_values":a.shared_answer_values.resolve(),"fitter":a.fitter.resolve(),"auditor":a.auditor.resolve(),"helper":a.helper.resolve()}
 for x in [*paths.values(),*(v.resolve() for v in a.terminal_scores)]:
  if not x.is_file():raise FileNotFoundError(x)
 sf,st,si=rj(paths["support_freeze"]),rj(paths["support_train_audit"]),rj(paths["support_internal_audit"])
 if st.get("decision")!="DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_TRAIN_OOF_GO" or si.get("decision")!="DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_INTERNAL_GO":raise ValueError("Corrected runtime support verifier is not GO")
 if st["input_hashes"]["freeze"]!=sh(paths["support_freeze"]):raise ValueError("Support train audit mismatch")
 pred_path=Path(st["output_paths"]["predictions"])
 if sh(pred_path)!=st["output_hashes"]["predictions"]:raise ValueError("Support predictions changed")
 support_pred={x["evidence_action_id"]:x for x in rjl(pred_path)}
 answer_scores={}
 for score_path in a.terminal_scores:
  for row in rjl(score_path.resolve()):
   if row["answer_action_id"] in answer_scores:raise ValueError("Duplicate terminal score")
   answer_scores[row["answer_action_id"]]=row
 terminal=rjl(paths["terminal_private"]);terminal_by_id={x["answer_action_id"]:x for x in terminal}
 shared={x["evidence_action_id"]:x for x in rjl(paths["shared_answer_values"])}
 if len(answer_scores)!=41273 or len(terminal_by_id)!=41273 or len(shared)!=14770:raise ValueError("Terminal universe mismatch")
 strategies=sorted({str(x.get("answer_strategy") or "unknown") for x in terminal})
 records=[];train_labels=[];internal_labels=[]
 for answer_id in sorted(answer_scores):
  label=terminal_by_id[answer_id];score=answer_scores[answer_id];evidence_id=label["evidence_action_id"];support=support_pred[evidence_id];strategy=str(label.get("answer_strategy") or "unknown")
  features={"support_logit":logit(float(support["semantic_support_probability"])),"reader_candidate_mean_logprob":float(score["reader_candidate_mean_logprob"]),"answer_token_length":float(score.get("answer_token_length") or 0),"is_unknown":float(bool(score.get("is_unknown")))}
  for value in strategies:features[f"strategy::{value}"]=float(strategy==value)
  records.append({"answer_action_id":answer_id,"evidence_action_id":evidence_id,"query_id":label["query_id"],"sample_id":label["sample_id"],"partition":label["partition"],"answer_strategy":strategy,"runtime_features":features})
  target={"answer_action_id":answer_id,"evidence_action_id":evidence_id,"sample_id":label["sample_id"],"partition":label["partition"],"corrected_support_label":None,"answer_correct_label":bool(label["answer_correct_proxy"]),"strict_success_label":None}
  # Support labels are physically split in the support freeze; load the matching partition below.
  (train_labels if label["partition"]=="policy_train" else internal_labels).append(target)
 support_label_paths={"policy_train":Path(sf["input_paths"]["private_train_labels"]),"internal_holdout":Path(sf["input_paths"]["private_internal_labels"])}
 support_labels={part:{x["evidence_action_id"]:bool(x["structured_support_label"]) for x in rjl(path)} for part,path in support_label_paths.items()}
 for group in (train_labels,internal_labels):
  for row in group:
   support=support_labels[row["partition"]][row["evidence_action_id"]];row["corrected_support_label"]=support;row["strict_success_label"]=bool(support and row["answer_correct_label"])
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);rp=out/"v6_terminal_value_full_v3_runtime_features_no_labels.jsonl";tp=out/"v6_terminal_value_full_v3_policy_train_labels_private.jsonl";ip=out/"v6_terminal_value_full_v3_internal_labels_private.jsonl"
 for path,rows in ((rp,records),(tp,train_labels),(ip,internal_labels)):
  with path.open("w") as h:
   for row in rows:h.write(json.dumps(row,sort_keys=True)+"\n")
 counts=Counter(x["partition"] for x in records);gates={"exact_41273_answer_actions":len(records)==41273,"exact_partition":counts==Counter({"policy_train":32955,"internal_holdout":8318}),"runtime_features_no_labels":all(not ({"gold","support_label","answer_correct","strict_success"}&set(x["runtime_features"])) for x in records),"corrected_support_used":True,"old_terminal_probability_not_used":True,"internal_labels_physical_separate":True,"dev_sealed":True,"test_sealed":True}
 protocol={"decision":"DAGIG_V6_TERMINAL_VALUE_FULL_V3_FROZEN" if all(gates.values()) else "DAGIG_V6_TERMINAL_VALUE_FULL_V3_NO_GO","protocol_version":"dagig_v6_corrected_support_no_gold_terminal_value_v3","feature_families":[["support_logit","reader_candidate_mean_logprob"],["support_logit","reader_candidate_mean_logprob","answer_token_length","is_unknown","support_x_reader"],["support_logit","reader_candidate_mean_logprob","answer_token_length","is_unknown","support_x_reader",*[f"strategy::{x}" for x in strategies]]],"fit":{"folds":5,"repeats":3,"l2_grid":[0.003,0.01,0.03,0.1,0.3],"steps":60,"seed":"dagig_v6_terminal_full_v3","clip":[1e-5,.99999]},"train_gates":{"auc_min":.88,"brier_improvement_min":.008,"pair_order_min":.65,"nonconstant_group_rate_min":.95,"support_and_reader_coefficients_positive":True},"internal_gates":{"auc_min":.87,"brier_improvement_min":.006,"pair_order_min":.62},"input_paths":{**{k:str(v) for k,v in paths.items()},"terminal_scores":[str(x.resolve()) for x in a.terminal_scores]},"input_hashes":{**{k:sh(v) for k,v in paths.items()},"terminal_scores":[sh(x.resolve()) for x in a.terminal_scores]},"output_paths":{"runtime_features":str(rp),"train_labels":str(tp),"internal_labels":str(ip)},"output_hashes":{"runtime_features":sh(rp),"train_labels":sh(tp),"internal_labels":sh(ip)},"gates":gates,"internal_used_for_fit":False,"dev_used":False,"test_used":False}
 fp=out/"DAGIG_V6_TERMINAL_VALUE_FULL_V3_FREEZE.json";fp.write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n");print(json.dumps({"decision":protocol["decision"],"counts":dict(counts),"gates":gates,"freeze":str(fp)},indent=2))
if __name__=="__main__":main()
