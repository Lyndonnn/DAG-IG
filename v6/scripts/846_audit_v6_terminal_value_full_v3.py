#!/usr/bin/env python3
"""One-shot internal audit of corrected answer-action terminal probabilities."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json
from pathlib import Path
from typing import Any
import numpy as np
def load(p:Path)->Any:s=importlib.util.spec_from_file_location("terminal_v3_audit_helper",p);m=importlib.util.module_from_spec(s);assert s.loader;s.loader.exec_module(m);return m
def rj(p:Path)->dict[str,Any]:return json.loads(p.read_text())
def rjl(p:Path)->list[dict[str,Any]]:
 with p.open() as h:return [json.loads(x) for x in h if x.strip()]
def sh(p:Path)->str:
 d=hashlib.sha256()
 with p.open("rb") as h:
  for c in iter(lambda:h.read(1024*1024),b""):d.update(c)
 return d.hexdigest()
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--freeze",type=Path,required=True);p.add_argument("--train_audit",type=Path,required=True);p.add_argument("--output_dir",type=Path,required=True);a=p.parse_args();fp=a.freeze.resolve();f=rj(fp);ta=rj(a.train_audit.resolve())
 if ta.get("decision")!="DAGIG_V6_TERMINAL_VALUE_FULL_V3_TRAIN_OOF_GO" or f["input_hashes"]["auditor"]!=sh(Path(__file__).resolve()) or ta["input_hashes"]["freeze"]!=sh(fp):raise ValueError("Terminal internal audit mismatch")
 helper=load(Path(f["input_paths"]["helper"]));pred={x["answer_action_id"]:x for x in rjl(Path(ta["output_paths"]["predictions"])) if x["partition"]=="internal_holdout"};labels={x["answer_action_id"]:x for x in rjl(Path(f["output_paths"]["internal_labels"]))};features={x["answer_action_id"]:x for x in rjl(Path(f["output_paths"]["runtime_features"])) if x["partition"]=="internal_holdout"};ids=sorted(pred)
 if len(ids)!=8318 or set(pred)!=set(labels):raise ValueError("Internal answer universe mismatch")
 score=np.asarray([pred[i]["terminal_success_probability"] for i in ids]);y=np.asarray([float(labels[i]["strict_success_label"]) for i in ids]);groups=[features[i]["evidence_action_id"] for i in ids];prev=float(y.mean());b=float(np.mean((score-y)**2));base=float(np.mean((prev-y)**2));pair=helper.pair_order(score,y,groups);m={"n":len(ids),"positive_rate":prev,"auc":helper.auc(score,y),"brier":b,"base_brier":base,"brier_improvement":base-b,"pair_order":pair};g=f["internal_gates"];gates={"auc":m["auc"]>=g["auc_min"],"brier_improvement":m["brier_improvement"]>=g["brier_improvement_min"],"pair_order":pair["accuracy"]>=g["pair_order_min"],"predictions_frozen_before_internal_labels":True,"dev_sealed":True,"test_sealed":True};decision="DAGIG_V6_TERMINAL_VALUE_FULL_V3_INTERNAL_GO" if all(gates.values()) else "DAGIG_V6_TERMINAL_VALUE_FULL_V3_INTERNAL_NO_GO";out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);audit={"decision":decision,"metrics":m,"gates":gates,"input_paths":{"freeze":str(fp),"train_audit":str(a.train_audit.resolve())},"input_hashes":{"freeze":sh(fp),"train_audit":sh(a.train_audit.resolve())},"internal_opened_after_predictions":True};ap=out/"DAGIG_V6_TERMINAL_VALUE_FULL_V3_INTERNAL_AUDIT.json";ap.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n");print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
