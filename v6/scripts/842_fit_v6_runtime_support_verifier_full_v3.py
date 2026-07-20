#!/usr/bin/env python3
"""Fit grouped-OOF no-gold support calibration on policy-train labels only."""

from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
import numpy as np

def load(path:Path)->Any:
    s=importlib.util.spec_from_file_location("dagig_runtime_support_fit_helper",path);m=importlib.util.module_from_spec(s);assert s.loader;s.loader.exec_module(m);return m
def read_json(path:Path)->dict[str,Any]:return json.loads(path.read_text(encoding="utf-8"))
def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open(encoding="utf-8") as h:return [json.loads(line) for line in h if line.strip()]
def sha256(path:Path)->str:
    d=hashlib.sha256()
    with path.open("rb") as h:
        for c in iter(lambda:h.read(1024*1024),b""):d.update(c)
    return d.hexdigest()
def matrix(names:list[str],records:list[dict[str,Any]],semantic:np.ndarray)->np.ndarray:
    cols=[]
    for name in names:
        if name=="semantic_logit":cols.append(semantic)
        else:cols.append(np.asarray([float(row["runtime_features"].get(name,0.0)) for row in records]))
    return np.stack(cols,axis=1)
def serialize(model:dict[str,np.ndarray])->dict[str,list[float]]:return {k:v.tolist() for k,v in model.items()}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--freeze",type=Path,required=True);p.add_argument("--score_dirs",type=Path,nargs="+",required=True);p.add_argument("--output_dir",type=Path,required=True);a=p.parse_args()
    fp=a.freeze.resolve();f=read_json(fp)
    if f.get("decision")!="DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_FROZEN" or f["input_hashes"]["fitter"]!=sha256(Path(__file__).resolve()):raise ValueError("Fitter/freeze mismatch")
    helper=load(Path(f["input_paths"]["helper"]));ip=Path(f["output_paths"]["verifier_inputs"]);records=sorted(read_jsonl(ip),key=lambda x:x["evidence_action_id"]);byid={r["evidence_action_id"]:r for r in records}
    scores={};man=[]
    for d in a.score_dirs:
        m=read_json(d.resolve()/"SHARD_MANIFEST.json");sp=Path(m["score_path"])
        if m.get("decision")!="DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_SHARD_COMPLETE" or m["freeze_sha256"]!=sha256(fp) or sha256(sp)!=m["score_sha256"]:raise ValueError(f"Bad score shard {d}")
        man.append(m)
        for row in read_jsonl(sp):
            if row["evidence_action_id"] in scores:raise ValueError("Duplicate score")
            scores[row["evidence_action_id"]]=row
    if set(scores)!=set(byid) or sorted(x["shard_index"] for x in man)!=list(range(man[0]["num_shards"])):raise ValueError("Score universe incomplete")
    label_path=Path(f["input_paths"]["private_train_labels"])
    if sha256(label_path)!=f["input_hashes"]["private_train_labels"]:raise ValueError("Policy-train support labels changed")
    labels_by_id={row["evidence_action_id"]:row for row in read_jsonl(label_path)}
    if any(row["partition"]!="policy_train" for row in labels_by_id.values()):raise ValueError("Fitter received non-train support labels")
    train=np.asarray([i for i,r in enumerate(records) if r["partition"]=="policy_train"]);internal=np.asarray([i for i,r in enumerate(records) if r["partition"]=="internal_holdout"])
    y=np.full(len(records),np.nan)
    for i in train:y[i]=float(labels_by_id[records[i]["evidence_action_id"]]["structured_support_label"])
    if not np.isfinite(y[train]).all():raise ValueError("Missing policy-train labels")
    semantic=np.asarray([scores[r["evidence_action_id"]]["semantic_support_logit"] for r in records]);config=f["fit"];gate=f["train_oof_gates"]
    prevalence=float(y[train].mean());base_brier=float(np.mean((prevalence-y[train])**2));samples=[records[i]["sample_id"] for i in train];queries=[records[i]["query_id"] for i in train]
    candidates=[];saved={}
    for family_index,names in enumerate(config["feature_families"]):
        x=matrix(names,records,semantic)
        for l2 in config["l2_grid"]:
            reps=[]
            for repeat in range(config["repeats"]):
                assign=helper.folds_for_samples(samples,config["folds"],f"{config['seed_prefix']}:{family_index}:{l2}:{repeat}");pred=np.full(len(train),np.nan)
                for fold in range(config["folds"]):
                    fit_idx=np.asarray([i for i in train if assign[records[i]["sample_id"]]!=fold]);val_idx=np.asarray([i for i in train if assign[records[i]["sample_id"]]==fold]);model=helper.fit_logistic(x[fit_idx],y[fit_idx],float(l2),config["newton_steps"]);pred[np.searchsorted(train,val_idx)]=helper.predict(model,x[val_idx])
                if not np.isfinite(pred).all():raise ValueError("Incomplete OOF")
                reps.append(pred)
            oof=np.mean(np.stack(reps),axis=0);full=helper.fit_logistic(x[train],y[train],float(l2),config["newton_steps"]);brier=float(np.mean((oof-y[train])**2));pair=helper.pair_order(oof,y[train],queries);groups=defaultdict(list)
            for pos,q in enumerate(queries):groups[q].append(pos)
            nonconstant=mean(float(max(oof[idx])-min(oof[idx])>1e-8) for idx in groups.values());semantic_positive=float(full["weights"][1])>0
            metric={"feature_family_index":family_index,"feature_names":names,"l2":float(l2),"auc":helper.auc(oof,y[train]),"brier":brier,"prevalence_brier":base_brier,"brier_improvement":base_brier-brier,"within_query_pair_order":pair,"nonconstant_query_rate":nonconstant,"semantic_coefficient":float(full["weights"][1])}
            passes=metric["auc"]>=gate["auc_min"] and metric["brier_improvement"]>=gate["brier_improvement_vs_prevalence_min"] and pair["accuracy"]>=gate["within_query_pair_order_min"] and nonconstant>=gate["nonconstant_query_rate_min"] and semantic_positive
            candidates.append({**metric,"passes":bool(passes)});saved[(family_index,float(l2))]=(oof,full,x)
    passing=[x for x in candidates if x["passes"]]
    selected=min((x for x in passing if x["feature_family_index"]==min(y["feature_family_index"] for y in passing)),key=lambda x:(x["brier"],x["l2"])) if passing else max(candidates,key=lambda x:(x["within_query_pair_order"]["accuracy"],x["auc"],-x["brier"]))
    oof,model,x=saved[(selected["feature_family_index"],selected["l2"])];low,high=config["probability_clip"];pred=np.full(len(records),np.nan);pred[train]=np.clip(oof,low,high);pred[internal]=np.clip(helper.predict(model,x[internal]),low,high)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);pp=out/"v6_runtime_support_values_full_v3_no_eval_labels.jsonl"
    with pp.open("w",encoding="utf-8") as h:
        for row,score,value in zip(records,semantic,pred):h.write(json.dumps({"evidence_action_id":row["evidence_action_id"],"query_id":row["query_id"],"sample_id":row["sample_id"],"partition":row["partition"],"semantic_support_logit":float(score),"semantic_support_probability":float(value),"prediction_source":"sample_group_oof" if row["partition"]=="policy_train" else "policy_train_full_fit"},sort_keys=True)+"\n")
    mp=out/"v6_runtime_support_calibrator_full_v3.json";mp.write_text(json.dumps({"feature_names":selected["feature_names"],"l2":selected["l2"],"model":serialize(model),"fit_partition":"policy_train_only"},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    gates={"candidate_passes":bool(passing),"complete_train_oof":len(train)==11795 and np.isfinite(pred[train]).all(),"complete_internal_predictions_without_labels":len(internal)==2975 and np.isfinite(pred[internal]).all(),"internal_labels_not_used":True,"dev_sealed":True,"test_sealed":True};decision="DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_TRAIN_OOF_GO" if all(gates.values()) else "DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_TRAIN_OOF_NO_GO"
    audit={"decision":decision,"selected_candidate":selected,"all_candidates":candidates,"gates":gates,"input_paths":{"freeze":str(fp),"score_dirs":[str(d.resolve()) for d in a.score_dirs]},"input_hashes":{"freeze":sha256(fp)},"output_paths":{"predictions":str(pp),"model":str(mp)},"output_hashes":{"predictions":sha256(pp),"model":sha256(mp)},"internal_labels_used":False,"dev_used":False,"test_used":False}
    ap=out/"DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_TRAIN_OOF_AUDIT.json";ap.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps({"decision":decision,"selected":selected,"gates":gates,"audit":str(ap)},indent=2))
if __name__=="__main__":main()
