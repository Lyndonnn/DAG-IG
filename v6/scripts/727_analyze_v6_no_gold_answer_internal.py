#!/usr/bin/env python3
"""Diagnose the sealed-internal DAG-vs-Outcome answer-node non-inferiority miss."""
from __future__ import annotations
import argparse,json,math,random,re
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean

def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def norm(x):return " ".join(re.findall(r"[\w]+",str(x or "").casefold(),re.UNICODE))
def percentile(values,q):
 values=sorted(values);position=(len(values)-1)*q;lo=int(position);hi=min(lo+1,len(values)-1);return values[lo]+(values[hi]-values[lo])*(position-lo)
def main():
 p=argparse.ArgumentParser();p.add_argument("--private_cases",type=Path,required=True);p.add_argument("--dagig_generations",type=Path,required=True);p.add_argument("--outcome_generations",type=Path,required=True);p.add_argument("--private_labels",type=Path,required=True);p.add_argument("--output_dir",type=Path,required=True);a=p.parse_args()
 cases=rows(a.private_cases);by_method=defaultdict(dict)
 for row in cases:by_method[row["method"]][row["parent_group_id"]]=row
 dag_gen={x["parent_group_id"]:x for x in rows(a.dagig_generations)};out_gen={x["parent_group_id"]:x for x in rows(a.outcome_generations)};labels={x["sample_id"]:x for x in rows(a.private_labels)}
 per_sample=defaultdict(lambda:{"groups":0,"dag_strict":0,"outcome_strict":0,"dag_only":0,"outcome_only":0})
 pair_rows=[];discord=Counter();same_answer=0
 for parent in sorted(by_method["dagig"]):
  d=by_method["dagig"][parent];o=by_method["outcome"][parent];dg=dag_gen[parent];og=out_gen[parent];sid=d["sample_id"];ds=int(d["generated_strict"]);os=int(o["generated_strict"])
  tag="both" if ds and os else "dagig_only" if ds else "outcome_only" if os else "neither";discord[tag]+=1;same_answer+=int(norm(dg["final_answer"])==norm(og["final_answer"]))
  s=per_sample[sid];s["groups"]+=1;s["dag_strict"]+=ds;s["outcome_strict"]+=os;s["dag_only"]+=int(tag=="dagig_only");s["outcome_only"]+=int(tag=="outcome_only")
  pair_rows.append({"sample_id":sid,"parent_group_id":parent,"gold_answer":labels[sid]["gold_answer"],"dagig_answer":dg["final_answer"],"outcome_answer":og["final_answer"],"dagig_strict":bool(ds),"outcome_strict":bool(os),"pair_type":tag,"evidence_supported":bool(d["evidence_supported"])})
 sample_diffs={sid:(x["dag_strict"]-x["outcome_strict"])/x["groups"] for sid,x in per_sample.items()};rng=random.Random(761943);ids=sorted(sample_diffs);boots=[]
 for _ in range(20000):boots.append(mean(sample_diffs[rng.choice(ids)] for _ in ids))
 b=discord["dagig_only"];c=discord["outcome_only"]
 # Two-sided exact sign/McNemar test over discordant groups.
 n=b+c;tail=sum(math.comb(n,k) for k in range(0,min(b,c)+1))/(2**n) if n else 1.0;pvalue=min(1.0,2*tail)
 summary={"groups":len(pair_rows),"samples":len(ids),"pair_counts":dict(discord),"same_generated_answer_rate":same_answer/len(pair_rows),"dagig_minus_outcome_group_strict":(discord["dagig_only"]-discord["outcome_only"])/len(pair_rows),"cluster_mean_sample_difference":mean(sample_diffs.values()),"cluster_bootstrap_95ci":[percentile(boots,.025),percentile(boots,.975)],"discordant_exact_p":pvalue,"samples_with_dagig_net_win":sum(v>0 for v in sample_diffs.values()),"samples_with_outcome_net_win":sum(v<0 for v in sample_diffs.values()),"samples_tied":sum(v==0 for v in sample_diffs.values()),"largest_outcome_concentrations":sorted(({"sample_id":sid,**x,"net":x["dag_strict"]-x["outcome_strict"]} for sid,x in per_sample.items()),key=lambda x:(x["net"],x["sample_id"]))[:10]}
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);(out/"v6_answer_dagig_vs_outcome_cases_private.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in pair_rows));(out/"DAGIG_V6_NO_GOLD_ANSWER_PAIRWISE_DIAGNOSIS.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
 lines=["# V6 Clean Answer DAG-IG vs Outcome Diagnosis","",f"- Groups/samples: `{summary['groups']} / {summary['samples']}`",f"- Pair counts: `{summary['pair_counts']}`",f"- Same generated answer rate: `{summary['same_generated_answer_rate']:.3f}`",f"- DAG minus Outcome strict: `{summary['dagig_minus_outcome_group_strict']:.4f}`",f"- Sample-cluster bootstrap 95% CI: `{summary['cluster_bootstrap_95ci']}`",f"- Discordant exact p: `{summary['discordant_exact_p']:.4f}`",f"- Sample wins/ties/losses: `{summary['samples_with_dagig_net_win']} / {summary['samples_tied']} / {summary['samples_with_outcome_net_win']}`","","## Concentration","",json.dumps(summary["largest_outcome_concentrations"],ensure_ascii=False,indent=2),""]
 (out/"DAGIG_V6_NO_GOLD_ANSWER_PAIRWISE_DIAGNOSIS.md").write_text("\n".join(lines));print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__":main()
