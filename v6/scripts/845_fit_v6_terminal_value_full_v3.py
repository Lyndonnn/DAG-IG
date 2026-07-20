#!/usr/bin/env python3
"""Grouped-OOF fit of corrected no-gold answer-action terminal values."""

from __future__ import annotations
import argparse,hashlib,importlib.util,json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
import numpy as np
def load(p:Path)->Any:s=importlib.util.spec_from_file_location("terminal_v3_helper",p);m=importlib.util.module_from_spec(s);assert s.loader;s.loader.exec_module(m);return m
def rj(p:Path)->dict[str,Any]:return json.loads(p.read_text())
def rjl(p:Path)->list[dict[str,Any]]:
 with p.open() as h:return [json.loads(x) for x in h if x.strip()]
def sh(p:Path)->str:
 d=hashlib.sha256()
 with p.open("rb") as h:
  for c in iter(lambda:h.read(1024*1024),b""):d.update(c)
 return d.hexdigest()
def matrix(names:list[str],rows:list[dict[str,Any]])->np.ndarray:
 cols=[]
 for name in names:
  if name=="support_x_reader":cols.append(np.asarray([x["runtime_features"]["support_logit"]*x["runtime_features"]["reader_candidate_mean_logprob"] for x in rows]))
  else:cols.append(np.asarray([x["runtime_features"].get(name,0.) for x in rows]))
 return np.stack(cols,axis=1)
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--freeze",type=Path,required=True);p.add_argument("--output_dir",type=Path,required=True);a=p.parse_args();fp=a.freeze.resolve();f=rj(fp)
 if f.get("decision")!="DAGIG_V6_TERMINAL_VALUE_FULL_V3_FROZEN" or f["input_hashes"]["fitter"]!=sh(Path(__file__).resolve()):raise ValueError("Terminal fitter mismatch")
 helper=load(Path(f["input_paths"]["helper"]));rows=rjl(Path(f["output_paths"]["runtime_features"]));labels={x["answer_action_id"]:x for x in rjl(Path(f["output_paths"]["train_labels"]))};train=np.asarray([i for i,x in enumerate(rows) if x["partition"]=="policy_train"]);internal=np.asarray([i for i,x in enumerate(rows) if x["partition"]=="internal_holdout"]);y=np.full(len(rows),np.nan)
 for i in train:y[i]=float(labels[rows[i]["answer_action_id"]]["strict_success_label"])
 cfg=f["fit"];gate=f["train_gates"];samples=[rows[i]["sample_id"] for i in train];groups=[rows[i]["evidence_action_id"] for i in train];prev=float(y[train].mean());base=float(np.mean((prev-y[train])**2));cands=[];saved={}
 for fi,names in enumerate(f["feature_families"]):
  x=matrix(names,rows)
  for l2 in cfg["l2_grid"]:
   repeats=[]
   for repeat in range(cfg["repeats"]):
    assign=helper.folds_for_samples(samples,cfg["folds"],f"{cfg['seed']}:{fi}:{l2}:{repeat}");pred=np.full(len(train),np.nan)
    for fold in range(cfg["folds"]):
     fit=np.asarray([i for i in train if assign[rows[i]["sample_id"]]!=fold]);val=np.asarray([i for i in train if assign[rows[i]["sample_id"]]==fold]);model=helper.fit_logistic(x[fit],y[fit],l2,cfg["steps"]);pred[np.searchsorted(train,val)]=helper.predict(model,x[val])
    repeats.append(pred)
   oof=np.mean(np.stack(repeats),axis=0);model=helper.fit_logistic(x[train],y[train],l2,cfg["steps"]);brier=float(np.mean((oof-y[train])**2));pair=helper.pair_order(oof,y[train],groups);d=defaultdict(list)
   for pos,g in enumerate(groups):d[g].append(pos)
   nonconstant=mean(float(max(oof[idx])-min(oof[idx])>1e-8) for idx in d.values());metric={"family":fi,"names":names,"l2":l2,"auc":helper.auc(oof,y[train]),"brier":brier,"base_brier":base,"brier_improvement":base-brier,"pair_order":pair,"nonconstant_group_rate":nonconstant,"support_coef":float(model["weights"][1]),"reader_coef":float(model["weights"][2])};passes=metric["auc"]>=gate["auc_min"] and metric["brier_improvement"]>=gate["brier_improvement_min"] and pair["accuracy"]>=gate["pair_order_min"] and nonconstant>=gate["nonconstant_group_rate_min"] and metric["support_coef"]>0 and metric["reader_coef"]>0;cands.append({**metric,"passes":bool(passes)});saved[(fi,l2)]=(oof,model,x)
 passing=[x for x in cands if x["passes"]];sel=min((x for x in passing if x["family"]==min(y["family"] for y in passing)),key=lambda x:(x["brier"],x["l2"])) if passing else max(cands,key=lambda x:(x["pair_order"]["accuracy"],x["auc"],-x["brier"]));oof,model,x=saved[(sel["family"],sel["l2"])];lo,hi=cfg["clip"];pred=np.full(len(rows),np.nan);pred[train]=np.clip(oof,lo,hi);pred[internal]=np.clip(helper.predict(model,x[internal]),lo,hi);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);pp=out/"v6_terminal_success_values_full_v3_no_eval_labels.jsonl"
 with pp.open("w") as h:
  for row,value in zip(rows,pred):h.write(json.dumps({"answer_action_id":row["answer_action_id"],"evidence_action_id":row["evidence_action_id"],"sample_id":row["sample_id"],"partition":row["partition"],"terminal_success_probability":float(value),"source":"sample_group_oof" if row["partition"]=="policy_train" else "policy_train_full_fit"},sort_keys=True)+"\n")
 gates={"candidate_passes":bool(passing),"complete_train_oof":np.isfinite(pred[train]).all(),"complete_internal_without_labels":np.isfinite(pred[internal]).all(),"internal_labels_not_read":True,"dev_sealed":True,"test_sealed":True};decision="DAGIG_V6_TERMINAL_VALUE_FULL_V3_TRAIN_OOF_GO" if all(gates.values()) else "DAGIG_V6_TERMINAL_VALUE_FULL_V3_TRAIN_OOF_NO_GO";audit={"decision":decision,"selected":sel,"candidates":cands,"gates":gates,"input_hashes":{"freeze":sh(fp)},"input_paths":{"freeze":str(fp)},"output_paths":{"predictions":str(pp)},"output_hashes":{"predictions":sh(pp)},"internal_labels_used":False};ap=out/"DAGIG_V6_TERMINAL_VALUE_FULL_V3_TRAIN_OOF_AUDIT.json";ap.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n");print(json.dumps({"decision":decision,"selected":sel,"gates":gates,"audit":str(ap)},indent=2))
if __name__=="__main__":main()
