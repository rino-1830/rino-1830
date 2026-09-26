import json, os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ["MODEL_ID"]; OUT=Path(os.environ["RESULT_PATH"])
OLD=192; RECENT=32; TARGET=16; N=24; SEED=101
torch.set_num_threads(min(4,os.cpu_count() or 1)); OUT.parent.mkdir(parents=True,exist_ok=True)

def legacy(c): return c.to_legacy_cache() if hasattr(c,"to_legacy_cache") else tuple(c)
def dyn(items):
    c=DynamicCache()
    for i,(k,v) in enumerate(items): c.update(k,v,i)
    return c
def clone(items): return [(k.clone(),v.clone()) for k,v in items]

def chunks(tok):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    docs=[];cur=[]
    for row in ds:
        t=row["text"];s=t.strip()
        if s.startswith("=") and s.endswith("=") and cur: docs.append("\n".join(cur));cur=[t]
        else: cur.append(t)
    if cur: docs.append("\n".join(cur))
    need=OLD+RECENT+TARGET;cands=[]
    for d in docs:
        ids=tok(d,add_special_tokens=False,return_tensors="pt").input_ids[0]
        if len(ids)>=need+8:cands.append(ids)
    r=random.Random(SEED);order=list(range(len(cands)));r.shuffle(order);out=[]
    for i in range(N):
        ids=cands[order[i%len(order)]];st=r.randrange(0,len(ids)-need+1);z=ids[st:st+need].unsqueeze(0)
        out.append((z[:,:OLD],z[:,OLD:OLD+RECENT],z[:,OLD+RECENT:]))
    return out

@torch.no_grad()
def prefill(model,x):
    o=model(x,use_cache=True,return_dict=True)
    return [(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]

@torch.no_grad()
def nll(model,items,recent,target):
    pos=torch.arange(OLD,OLD+recent.shape[1]);o=model(recent,past_key_values=dyn(clone(items)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    logits=[o.logits[:,-1:,:].float()]
    cache=[(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]
    if target.shape[1]>1:
        inp=target[:,:-1];p2=torch.arange(OLD+RECENT,OLD+RECENT+inp.shape[1])
        q=model(inp,past_key_values=dyn(cache),position_ids=p2.unsqueeze(0),cache_position=p2,use_cache=False,return_dict=True)
        logits.append(q.logits.float())
    lg=torch.cat(logits,1)[:,:target.shape[1]]
    return float(F.cross_entropy(lg.reshape(-1,lg.shape[-1]),target.reshape(-1)))

def main():
    st=time.time();tok=AutoTokenizer.from_pretrained(MODEL_ID);model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True);model.eval()
    rows=[]
    for i,(old,recent,target) in enumerate(chunks(tok)):
        items=prefill(model,old)
        full=nll(model,items,recent,target)
        anchor=[(k[...,-1:,:].clone(),v[...,-1:,:].clone()) for k,v in items]
        short=nll(model,anchor,recent,target)
        rows.append({"full":full,"recent_only":short,"delta_full_minus_recent":full-short})
        print(i,round(full,3),round(short,3),flush=True)
    mean=lambda k:sum(r[k] for r in rows)/len(rows)
    result={"model":MODEL_ID,"dataset":"WikiText-2 validation within-document","windows":N,"old":OLD,"recent":RECENT,"target":TARGET,
            "full_nll":mean("full"),"recent_nll":mean("recent_only"),"full_minus_recent":mean("delta_full_minus_recent"),
            "recent_wins":sum(r["recent_only"]<r["full"] for r in rows),"rows":rows,"elapsed":time.time()-st}
    OUT.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ["full_nll","recent_nll","full_minus_recent","recent_wins"]}),flush=True)
if __name__=="__main__":main()
