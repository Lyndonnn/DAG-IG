#!/usr/bin/env python3
"""Score one GPU shard of the frozen no-gold support verifier."""

from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration

def read_json(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding="utf-8"))
def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open(encoding="utf-8") as h:return [json.loads(line) for line in h if line.strip()]
def sha256(path:Path)->str:
    d=hashlib.sha256()
    with path.open("rb") as h:
        for c in iter(lambda:h.read(1024*1024),b""):d.update(c)
    return d.hexdigest()
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--freeze",type=Path,required=True);p.add_argument("--output_dir",type=Path,required=True);p.add_argument("--shard_index",type=int,required=True);p.add_argument("--num_shards",type=int,required=True);p.add_argument("--batch_size",type=int,default=8);a=p.parse_args()
    fpath=a.freeze.resolve();f=read_json(fpath)
    if f.get("decision")!="DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_FROZEN" or f["input_hashes"]["scorer"]!=sha256(Path(__file__).resolve()):raise ValueError("Runtime verifier protocol/scorer mismatch")
    if a.num_shards!=int(f["verifier"]["num_shards"]) or not 0<=a.shard_index<a.num_shards:raise ValueError("Shard mismatch")
    ipath=Path(f["output_paths"]["verifier_inputs"])
    if sha256(ipath)!=f["output_hashes"]["verifier_inputs"]:raise ValueError("Verifier inputs changed")
    rows=[row for i,row in enumerate(sorted(read_jsonl(ipath),key=lambda x:x["evidence_action_id"])) if i%a.num_shards==a.shard_index]
    root=Path(f["model_fingerprint"]["path"])
    for name,expected in f["model_fingerprint"]["files"].items():
        path=root/name
        if not path.is_file() or path.stat().st_size!=expected["bytes"] or sha256(path)!=expected["sha256"]:raise ValueError(f"Model changed: {path}")
    proc=AutoProcessor.from_pretrained(root,local_files_only=True);tok=proc.tokenizer;tok.padding_side="left";tok.pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids={label:tok.encode(label,add_special_tokens=False)[0] for label in ("A","B")}
    if any(len(tok.encode(label,add_special_tokens=False))!=1 for label in ids):raise ValueError("A/B labels not one token")
    texts=[];counts=[]
    for row in rows:
        text=tok.apply_chat_template([{"role":"system","content":row["system_prompt"]},{"role":"user","content":row["user_prompt"]}],tokenize=False,add_generation_prompt=True);count=len(tok.encode(text,add_special_tokens=False))
        if count>int(f["verifier"]["max_input_tokens"]):raise ValueError(f"Prompt too long: {row['evidence_action_id']}={count}")
        texts.append(text);counts.append(count)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(root,torch_dtype=torch.bfloat16,attn_implementation=f["verifier"]["attn_implementation"],local_files_only=True).eval().cuda()
    scored=[]
    with torch.inference_mode():
        for start in range(0,len(rows),a.batch_size):
            enc=tok(texts[start:start+a.batch_size],return_tensors="pt",padding=True,truncation=False).to("cuda"); logits=model(**enc).logits[:,-1,:].float(); pair=logits[:,[ids["A"],ids["B"]]]; probs=torch.softmax(pair,dim=-1)[:,0].tolist();diff=(pair[:,0]-pair[:,1]).tolist()
            for row,pr,lg,count in zip(rows[start:start+a.batch_size],probs,diff,counts[start:start+a.batch_size]):
                if not math.isfinite(pr) or not math.isfinite(lg):raise ValueError("Nonfinite verifier score")
                scored.append({"evidence_action_id":row["evidence_action_id"],"query_id":row["query_id"],"sample_id":row["sample_id"],"partition":row["partition"],"semantic_support_logit":float(lg),"semantic_support_raw_probability":float(pr),"input_token_count":count})
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);sp=out/f"v6_runtime_support_scores_shard{a.shard_index:02d}_of_{a.num_shards:02d}.jsonl"
    with sp.open("w",encoding="utf-8") as h:
        for row in scored:h.write(json.dumps(row,sort_keys=True)+"\n")
    m={"decision":"DAGIG_V6_RUNTIME_SUPPORT_VERIFIER_FULL_V3_SHARD_COMPLETE","freeze_sha256":sha256(fpath),"shard_index":a.shard_index,"num_shards":a.num_shards,"rows":len(scored),"score_path":str(sp),"score_sha256":sha256(sp),"private_labels_loaded":False,"dev_used":False,"test_used":False,"api_calls":0}
    mp=out/"SHARD_MANIFEST.json";mp.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps(m,indent=2))
if __name__=="__main__":main()
