#!/usr/bin/env python3
"""Score one clean answer policy on a frozen no-label action partition."""

from __future__ import annotations

import argparse, gc, hashlib, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def read_jsonl(path: Path) -> list[dict[str, Any]]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def field_tokens(tokenizer: Any, completion: str) -> tuple[list[int], list[int]]:
    value=json.loads(completion)["final_answer"]; marker=json.dumps("final_answer")+":"; start=completion.find(marker)+len(marker)
    serialized=json.dumps(value,ensure_ascii=False,separators=(",",":")); end=start+len(serialized)
    encoded=tokenizer(completion,add_special_tokens=False,return_offsets_mapping=True)
    mask=[int(right>start and left<end) for left,right in encoded["offset_mapping"]]
    if not any(mask): raise ValueError("empty final_answer mask")
    return list(encoded["input_ids"]),mask


def score_group(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_tokens: int) -> list[float]:
    prefix=tokenizer.apply_chat_template([{"role":"user","content":rows[0]["prompt"]}],tokenize=True,add_generation_prompt=True)
    sequences=[]; masks=[]
    for row in rows:
        ids,field_mask=field_tokens(tokenizer,row["completion"]); seq=prefix+ids+[tokenizer.eos_token_id]; mask=[0]*len(prefix)+field_mask+[0]
        if len(seq)>max_tokens: raise ValueError(f"score sequence too long: {len(seq)}")
        sequences.append(seq); masks.append(mask)
    width=max(map(len,sequences)); input_ids=torch.full((len(sequences),width),tokenizer.pad_token_id,dtype=torch.long); attention=torch.zeros_like(input_ids); field_masks=torch.zeros((len(sequences),width))
    for i,(seq,mask) in enumerate(zip(sequences,masks)):
        input_ids[i,:len(seq)]=torch.tensor(seq); attention[i,:len(seq)]=1; field_masks[i,:len(seq)]=torch.tensor(mask)
    with torch.inference_mode(): logits=model(input_ids=input_ids.cuda(),attention_mask=attention.cuda(),use_cache=False).logits[:,:-1].float()
    labels=input_ids[:,1:].cuda(); token_logp=torch.log_softmax(logits,dim=-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1); mask=field_masks[:,1:].cuda()
    return ((token_logp*mask).sum(-1)/mask.sum(-1).clamp_min(1.0)).cpu().tolist()


def parse_answer(text: str) -> tuple[str,bool]:
    raw=str(text or "").strip(); decoder=json.JSONDecoder()
    for index,char in enumerate(raw):
        if char!="{": continue
        try:value,_=decoder.raw_decode(raw[index:])
        except json.JSONDecodeError: continue
        if isinstance(value,dict) and set(value)=={"final_answer"} and isinstance(value["final_answer"],str):
            answer=" ".join(value["final_answer"].split()); return answer or "unknown",bool(answer)
    return "unknown",False


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--freeze",type=Path,required=True); p.add_argument("--train_audit",type=Path,required=True)
    p.add_argument("--method",choices=("no_credit","local_ig","outcome","dagig"),required=True); p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--partition",choices=("policy_train","internal_holdout"),required=True); p.add_argument("--generate",action="store_true"); p.add_argument("--output_dir",type=Path,required=True)
    args=p.parse_args(); freeze_path=args.freeze.resolve(); audit_path=args.train_audit.resolve(); freeze=read_json(freeze_path); train=read_json(audit_path)
    if freeze.get("decision")!="DAGIG_V6_NO_GOLD_ANSWER_CONTROLS_FROZEN" or train.get("decision")!="DAGIG_V6_NO_GOLD_ANSWER_POLICY_READY": raise ValueError("answer freeze/train not ready")
    if train.get("method")!=args.method or sha256(args.adapter.resolve()/"adapter_model.safetensors")!=train["output_hashes"]["adapter_model"]: raise ValueError("method adapter mismatch")
    data_key="train_data" if args.partition=="policy_train" else "internal_data"; grouped:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for row in read_jsonl(Path(freeze["input_paths"][data_key])): grouped[str(row["parent_group_id"])].append(row)
    groups=[sorted(rows,key=lambda row:row["answer_action_id"]) for _,rows in sorted(grouped.items())]
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration
    tokenizer=AutoTokenizer.from_pretrained(freeze["base_model"],local_files_only=True); tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token; tokenizer.padding_side="left"
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(freeze["base_model"],torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True)
    model=PeftModel.from_pretrained(base,args.adapter.resolve()).cuda().eval(); score_rows=[]
    for index,rows in enumerate(groups):
        scores=score_group(model,tokenizer,rows,int(freeze["training"]["max_input_tokens"])); probabilities=torch.softmax(torch.tensor(scores,dtype=torch.float64),dim=0).tolist()
        score_rows.append({"method":args.method,"partition":args.partition,"sample_id":rows[0]["sample_id"],"parent_group_id":rows[0]["parent_group_id"],"action_ids":[row["answer_action_id"] for row in rows],"field_logprob_scores":scores,"policy_probabilities":probabilities})
        if (index+1)%400==0: print(json.dumps({"method":args.method,"scored":index+1,"total":len(groups)}),flush=True)
    generations=[]
    if args.generate:
        batch_size=4
        for start in range(0,len(groups),batch_size):
            batch_groups=groups[start:start+batch_size]; rendered=[tokenizer.apply_chat_template([{"role":"user","content":rows[0]["prompt"]}],tokenize=False,add_generation_prompt=True) for rows in batch_groups]
            encoded=tokenizer(rendered,padding=True,return_tensors="pt"); lengths=encoded["attention_mask"].sum(1).tolist()
            if max(lengths)>int(freeze["training"]["max_input_tokens"]): raise ValueError("generation prompt too long")
            inputs={k:v.cuda() for k,v in encoded.items()}
            with torch.inference_mode(): generated=model.generate(**inputs,do_sample=False,max_new_tokens=64,pad_token_id=tokenizer.pad_token_id)
            texts=tokenizer.batch_decode(generated[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            for rows,raw in zip(batch_groups,texts):
                answer,valid=parse_answer(raw); generations.append({"method":args.method,"partition":args.partition,"sample_id":rows[0]["sample_id"],"parent_group_id":rows[0]["parent_group_id"],"raw_generation":raw.strip(),"final_answer":answer,"valid_json":valid})
            if min(start+batch_size,len(groups))%400==0: print(json.dumps({"method":args.method,"generated":min(start+batch_size,len(groups)),"total":len(groups)}),flush=True)
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False); scores_path=output/f"v6_answer_{args.method}_{args.partition}_scores_no_labels.jsonl"; scores_path.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in score_rows),encoding="utf-8")
    paths={"scores":str(scores_path)}; hashes={"scores":sha256(scores_path)}
    if args.generate:
        gen_path=output/f"v6_answer_{args.method}_{args.partition}_generations_no_labels.jsonl"; gen_path.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in generations),encoding="utf-8"); paths["generations"]=str(gen_path); hashes["generations"]=sha256(gen_path)
    ready={"decision":"DAGIG_V6_NO_GOLD_ANSWER_POLICY_SCORES_READY","method":args.method,"partition":args.partition,"metrics":{"groups":len(groups),"action_rows":sum(map(len,groups)),"generated":len(generations)},"gates":{"finite_scores":all(math.isfinite(v) for row in score_rows for v in [*row["field_logprob_scores"],*row["policy_probabilities"]]),"normalized_policies":all(abs(sum(row["policy_probabilities"])-1)<1e-8 for row in score_rows),"no_gold_or_qrels_loaded":True,"internal_holdout_unused_for_training":True,"dev_sealed":True,"test_sealed":True},"output_paths":paths,"output_hashes":hashes,"gold_or_qrels_loaded":False,"training_run":False}
    (output/"DAGIG_V6_NO_GOLD_ANSWER_POLICY_SCORE_AUDIT.json").write_text(json.dumps(ready,indent=2,sort_keys=True)+"\n"); print(json.dumps(ready,indent=2,sort_keys=True)); del model,base; gc.collect(); torch.cuda.empty_cache()


if __name__=="__main__": main()
