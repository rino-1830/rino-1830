import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ.get("MODEL_ID","prism-ml/Bonsai-1.7B-unpacked")
OUT=Path(os.environ.get("RESULT_PATH","boundary-compression-lab/results/oracle_kv_results.json"))
RECENT=int(os.environ.get("RECENT_TOKENS","48"))
MAX_PREFIX=int(os.environ.get("MAX_PREFIX","192"))
TEST_CASES=int(os.environ.get("TEST_CASES","8"))
OUT.parent.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(min(4,os.cpu_count() or 1))
random.seed(19)

def make_code(i):
    left=["LANTERN","ORBIT","CEDAR","MICA","NOVA","RAVEN","EMBER","LOTUS","QUARTZ","MAPLE","SOLAR","VIOLET","COMET","FROST","IVORY","DELTA","SABLE","CORAL","ONYX","AURORA","PINE","AMBER","FLINT","IRIS","LUNAR","BIRCH","OPAL","TERRA","ZEPHYR","CINDER","JADE","KITE"]
    return f"{left[i%len(left)]}-{4100+((i*379)%5800):04d}"

def filler(seed,repeats=10):
    variants=[
        "A field notebook records routine observations about tools, roads, materials, weather, and schedules. ",
        "The maintenance log lists ordinary checks, measurements, replacement parts, and inspection notes. ",
        "Unrelated notes describe storage rooms, work benches, cables, labels, and daily operating procedures. ",
        "The report contains mundane details that should not replace the identifiers stored in the table. ",
    ]
    r=random.Random(seed)
    return "".join(r.choice(variants) for _ in range(repeats))

def make_case(case_id,q):
    labels=["ALPHA","BETA","GAMMA","DELTA"]
    codes=[make_code(case_id*4+j) for j in range(4)]
    table="\n".join(f"{labels[j]} = {codes[j]}" for j in range(4))
    prefix="Memorize the following identifier table exactly.\n"+table+"\n\n"+filler(case_id)+f"\nQuestion: What is the exact identifier for {labels[q]}?\nAnswer:"
    return prefix," "+codes[q],{"case_id":case_id,"query":labels[q],"answer":codes[q]}

def legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache,"to_legacy_cache") else tuple(cache)

def from_legacy(items):
    if hasattr(DynamicCache,"from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(items))
    c=DynamicCache()
    for i,(k,v) in enumerate(items): c.update(k,v,i)
    return c

def subset_cache(cache,idx):
    return from_legacy([(k.index_select(-2,idx).contiguous(),v.index_select(-2,idx).contiguous()) for k,v in legacy(cache)])

def pick_indices(n,old_len,recent,kind,k=None):
    device=torch.device("cpu")
    recent_idx=torch.arange(old_len,n,device=device)
    if kind=="full": return torch.arange(n,device=device)
    if kind=="recent": return recent_idx
    k=min(k or 0,old_len)
    if kind=="first":
        old=torch.arange(0,k,device=device)
    elif kind=="last_old":
        old=torch.arange(old_len-k,old_len,device=device)
    elif kind=="uniform":
        if k<=0: old=torch.empty(0,dtype=torch.long)
        else: old=torch.linspace(0,old_len-1,k,device=device).round().long().unique()
    else: raise ValueError(kind)
    return torch.cat([old,recent_idx]).unique(sorted=True)

@torch.inference_mode()
def score(model,tok,prefix,target,methods):
    p=tok(prefix,add_special_tokens=False,return_tensors="pt").input_ids
    t=tok(target,add_special_tokens=False,return_tensors="pt").input_ids[:,:12]
    if p.shape[1]>MAX_PREFIX:
        p=torch.cat([p[:,:MAX_PREFIX-RECENT],p[:,-RECENT:]],dim=1)
    old_len=p.shape[1]-RECENT
    pre=model(p,use_cache=True,return_dict=True)
    cache=pre.past_key_values
    # First target token is predicted before pruning and is therefore common to all methods.
    first_pred=pre.logits[:,-1,:].argmax(dim=-1)
    rows=[]
    for name,kind,k in methods:
        idx=pick_indices(p.shape[1],old_len,RECENT,kind,k)
        c=subset_cache(cache,idx)
        correct=int((first_pred==t[:,0]).item())
        losses=[F.cross_entropy(pre.logits[:,-1,:].float(),t[:,0]).item()]
        preds=[int(first_pred.item())]
        for j in range(t.shape[1]-1):
            pos=p.shape[1]+j
            out=model(t[:,j:j+1],past_key_values=c,use_cache=True,
                      cache_position=torch.tensor([pos]),position_ids=torch.tensor([[pos]]),return_dict=True)
            c=out.past_key_values
            pred=out.logits[:,-1,:].argmax(dim=-1)
            correct+=int((pred==t[:,j+1]).item())
            preds.append(int(pred.item()))
            losses.append(F.cross_entropy(out.logits[:,-1,:].float(),t[:,j+1]).item())
        rows.append({
            "method":name,"retained_prefix_slots":int(idx.numel()),
            "retained_old_slots":int((idx<old_len).sum().item()),
            "old_len":int(old_len),"target_tokens":int(t.shape[1]),
            "nll":sum(losses)/len(losses),
            "token_accuracy":correct/int(t.shape[1]),
            "exact":correct==int(t.shape[1]),
            "prediction":tok.decode(preds),
        })
    return {"prefix_tokens":int(p.shape[1]),"old_tokens":int(old_len),"target":target,"rows":rows}

def main():
    started=time.time()
    print("loading",MODEL_ID,flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    methods=[("full","full",None),("recent48","recent",None)]
    for k in [8,16,24,32,40,48,64,96]:
        methods.append((f"first{k}+recent","first",k))
    for k in [16,32,48]:
        methods.append((f"uniform{k}+recent","uniform",k))
    cases=[]
    for i in range(TEST_CASES):
        p,t,m=make_case(60+i,i%4)
        print("case",m,flush=True)
        r=score(model,tok,p,t,methods); r["meta"]=m; cases.append(r)
        for row in r["rows"]:
            print(row["method"],"exact",row["exact"],"acc",round(row["token_accuracy"],3),"nll",round(row["nll"],3),flush=True)
    summary={}
    for name,_,_ in methods:
        rs=[next(x for x in c["rows"] if x["method"]==name) for c in cases]
        summary[name]={
            "exact_rate":sum(x["exact"] for x in rs)/len(rs),
            "token_accuracy":sum(x["token_accuracy"] for x in rs)/len(rs),
            "avg_nll":sum(x["nll"] for x in rs)/len(rs),
            "avg_retained_prefix_slots":sum(x["retained_prefix_slots"] for x in rs)/len(rs),
        }
    out={"model":MODEL_ID,"recent_tokens":RECENT,"max_prefix":MAX_PREFIX,"cases":cases,"summary":summary,"elapsed_seconds":time.time()-started}
    OUT.write_text(json.dumps(out,indent=2))
    print("SUMMARY",json.dumps(summary,indent=2),flush=True)
    print("wrote",OUT,flush=True)

if __name__=="__main__": main()
