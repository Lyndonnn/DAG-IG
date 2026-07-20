#!/usr/bin/env python3
"""One-shot internal audit of frozen no-gold runtime support predictions."""

from __future__ import annotations
import argparse,hashlib,importlib.util,json
from pathlib import Path
from typing import Any
import numpy as np
def load(path:Path)->Any:
 s=importlib.util.spec_from_file_location("dagig_runtime_support_audit_helper",path);m=importlib.util.module_from_spec(s);assert s.loader;s.loader.exec_module(m);return m
def rj(path:Path)->dict[str,Any]:return json.loads(path.read_text())
def rjl(path:Path)->list[dict[str,Any]]:
 with path.open() as h:return [json.loads(x) for x in h if x.strip()]
def sh(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(1024*1024),b""):d.update(c)
 return d.hexdigest()
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--freeze",type=Path,required=True);p.add_argument("--train_audit",type=Path,required=True);p.add_argument("--output_dir",type=Path,required=True);a=p.parse_args();fp=a.freeze.resolve();f=rj(fp);ta=rj(a.train_audit.resolve())
 if ta.get("decision")!="DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_TRAIN_OOF_GO" or f["input_hashes"]["auditor"]!=sh(Path(__file__).resolve()) or ta["input_hashes"]["freeze"]!=sh(fp):raise ValueError("Runtime verifier not frozen/GO")
 helper=load(Path(f["input_paths"]["helper"]));pred={x["evidence_action_id"]:x for x in rjl(Path(ta["output_paths"]["predictions"])) if x["partition"]=="internal_holdout"};internal_path=Path(f["input_paths"]["private_internal_labels"])
 if sh(internal_path)!=f["input_hashes"]["private_internal_labels"]:raise ValueError("Internal support labels changed")
 labels={x["evidence_action_id"]:x for x in rjl(internal_path)};inputs={x["evidence_action_id"]:x for x in rjl(Path(f["output_paths"]["verifier_inputs"])) if x["partition"]=="internal_holdout"}
 if len(pred)!=len(labels) or len(pred)!=2975 or set(pred)!=set(labels):raise ValueError("Internal universe mismatch")
 ids=sorted(pred);scores=np.asarray([pred[i]["semantic_support_probability"] for i in ids]);y=np.asarray([float(labels[i]["structured_support_label"]) for i in ids]);queries=[inputs[i]["query_id"] for i in ids];prev=float(y.mean());brier=float(np.mean((scores-y)**2));base=float(np.mean((prev-y)**2));pair=helper.pair_order(scores,y,queries);metrics={"n":len(ids),"positive_rate":prev,"auc":helper.auc(scores,y),"brier":brier,"prevalence_brier":base,"brier_improvement":base-brier,"within_query_pair_order":pair};g=f["internal_gates"];gates={"auc":metrics["auc"]>=g["auc_min"],"brier_improvement":metrics["brier_improvement"]>=g["brier_improvement_vs_prevalence_min"],"within_query_pair_order":pair["accuracy"]>=g["within_query_pair_order_min"],"predictions_frozen_before_internal_labels":True,"dev_sealed":True,"test_sealed":True};decision="DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_INTERNAL_GO" if all(gates.values()) else "DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_INTERNAL_NO_GO";out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);audit={"decision":decision,"metrics":metrics,"gates":gates,"input_paths":{"freeze":str(fp),"train_audit":str(a.train_audit.resolve())},"input_hashes":{"freeze":sh(fp),"train_audit":sh(a.train_audit.resolve())},"internal_labels_opened_after_predictions_frozen":True,"dev_used":False,"test_used":False};ap=out/"DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_INTERNAL_AUDIT.json";ap.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n");print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
