import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = os.environ.get("MODEL_ID", "prism-ml/Bonsai-1.7B-unpacked")
OUT = Path(os.environ.get("RESULT_PATH", "boundary-compression-lab/results/learned_kv_selector_results.json"))
CKPT = Path(os.environ.get("CKPT_PATH", "boundary-compression-lab/results/learned_kv_selector.pt"))
SLOTS = int(os.environ.get("BOUNDARY_SLOTS", "16"))
RECENT = int(os.environ.get("RECENT_TOKENS", "48"))
MAX_PREFIX = int(os.environ.get("MAX_PREFIX", "192"))
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "96"))
LR = float(os.environ.get("LR", "0.01"))
RANK = int(os.environ.get("BOUNDARY_RANK", "64"))
SEED = 11

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 1))

def make_code(i):
    left = ["LANTERN","ORBIT","CEDAR","MICA","NOVA","RAVEN","EMBER","LOTUS",
            "QUARTZ","MAPLE","SOLAR","VIOLET","COMET","FROST","IVORY","DELTA",
            "SABLE","CORAL","ONYX","AURORA","PINE","AMBER","FLINT","IRIS",
            "LUNAR","BIRCH","OPAL","TERRA","ZEPHYR","CINDER","JADE","KITE"]
    return f"{left[i % len(left)]}-{4100 + ((i * 379) % 5800):04d}"

def filler(seed, repeats=10):
    variants=[
      "A field notebook records routine observations about tools, roads, materials, weather, and schedules. ",
      "The maintenance log lists ordinary checks, measurements, replacement parts, and inspection notes. ",
      "Unrelated notes describe storage rooms, work benches, cables, labels, and daily operating procedures. ",
      "The report contains mundane details that should not replace the identifiers stored in the table. ",
    ]
    r=random.Random(seed)
    return "".join(r.choice(variants) for _ in range(repeats))

def make_case(case_id, query_slot):
    labels=["ALPHA","BETA","GAMMA","DELTA"]
    codes=[make_code(case_id*4+j) for j in range(4)]
    table="\n".join(f"{labels[j]} = {codes[j]}" for j in range(4))
    prefix=("Memorize the following identifier table exactly.\n"+table+"\n\n"+
            filler(case_id)+
            f"\nQuestion: What is the exact identifier for {labels[query_slot]}?\nAnswer:")
    return prefix, " "+codes[query_slot], {"case_id":case_id,"query":labels[query_slot],"answer":codes[query_slot]}

def natural_case():
    p=("A transformer language model predicts the next token from a sequence of earlier tokens. "
       "During autoregressive inference, attention keys and values from earlier positions are cached "
       "so they do not need to be recomputed. As the context grows, this cache can become a large "
       "fraction of total memory use. One possible strategy is to preserve only the information from "
       "the distant prefix that is useful for predicting the continuation. In that setting, the main "
       "question is whether a small boundary state can replace most of the old cache without changing ")
    return p, "the model prediction very much."

def encode_pair(tok,prefix,target):
    p=tok(prefix,add_special_tokens=False,return_tensors="pt").input_ids
    t=tok(target,add_special_tokens=False,return_tensors="pt").input_ids[:, :12]
    if p.shape[1]>MAX_PREFIX:
        left=MAX_PREFIX-RECENT
        p=torch.cat([p[:,:left],p[:,-RECENT:]],dim=1)
    return p,t

def legacy(cache):
    if hasattr(cache,"to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(cache)

def dyn(items):
    if hasattr(DynamicCache,"from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(items))
    c=DynamicCache()
    for i,(k,v) in enumerate(items):
        c.update(k,v,i)
    return c

class KVSelector(nn.Module):
    def __init__(self, hidden, slots=SLOTS, rank=RANK):
        super().__init__()
        self.key_proj=nn.Linear(hidden,rank,bias=False)
        self.queries=nn.Parameter(torch.randn(slots,rank)/math.sqrt(rank))
        self.log_temp=nn.Parameter(torch.tensor(math.log(0.7)))

    def soft_weights(self, contextual):
        k=F.normalize(self.key_proj(contextual.float()),dim=-1)
        q=F.normalize(self.queries,dim=-1)
        temp=self.log_temp.exp().clamp(0.08,4.0)
        score=torch.einsum("sr,bnr->bsn",q,k)/temp
        return score.softmax(dim=-1)

    def forward(self, contextual):
        soft=self.soft_weights(contextual)[0]  # [S,N]
        # Unique hard choices for the forward pass, soft weights for the backward pass.
        score=soft.detach().clone()
        chosen=[]
        used=torch.zeros(score.shape[-1],dtype=torch.bool,device=score.device)
        for s in range(score.shape[0]):
            masked=score[s].masked_fill(used,-1.0)
            idx=int(masked.argmax().item())
            chosen.append(idx)
            used[idx]=True
        hard=F.one_hot(torch.tensor(chosen,device=soft.device),num_classes=soft.shape[-1]).to(soft.dtype)
        # Keep cache entries in original chronological order.
        order=torch.argsort(torch.tensor(chosen,device=soft.device))
        hard=hard.index_select(0,order)
        soft_ord=soft.index_select(0,order)
        st=hard + soft_ord - soft_ord.detach()
        positions=torch.tensor(chosen,device=soft.device).index_select(0,order)
        return st, soft_ord, positions

@torch.no_grad()
def cache_features(model,tok,prefix,target):
    p,t=encode_pair(tok,prefix,target)
    # Leave the final prefix token out of cache; it will be fed as the first scoring token.
    cached_ids=p[:,:-1]
    last_prefix=p[:,-1:]
    old_len=max(1,cached_ids.shape[1]-RECENT)
    out=model(cached_ids,use_cache=True,output_hidden_states=True,return_dict=True)
    full=[(k.detach().float().clone(),v.detach().float().clone()) for k,v in legacy(out.past_key_values)]
    contextual=out.hidden_states[-1][:,:old_len,:].detach().float().clone()
    return {"prefix_ids":p,"target_ids":t,"last_prefix":last_prefix,"old_len":old_len,
            "cached_len":cached_ids.shape[1],"full_cache":full,"contextual_old":contextual}

def build_selected_cache(item, st):
    old_len=item["old_len"]
    items=[]
    for k,v in item["full_cache"]:
        oldk=k[...,:old_len,:]
        oldv=v[...,:old_len,:]
        recentk=k[...,old_len:,:]
        recentv=v[...,old_len:,:]
        ck=torch.einsum("sn,bhnd->bhsd",st,oldk)
        cv=torch.einsum("sn,bhnd->bhsd",st,oldv)
        items.append((torch.cat([ck,recentk],dim=-2),torch.cat([cv,recentv],dim=-2)))
    return dyn(items)

def build_full_cache(item):
    return dyn([(k.clone(),v.clone()) for k,v in item["full_cache"]])

def build_recent_cache(item):
    old=item["old_len"]
    return dyn([(k[...,old:,:].clone(),v[...,old:,:].clone()) for k,v in item["full_cache"]])

def score_with_cache(model,item,cache):
    t=item["target_ids"]
    seq=torch.cat([item["last_prefix"],t[:,:-1]],dim=1)
    start=item["cached_len"]
    pos=torch.arange(start,start+seq.shape[1],dtype=torch.long).unsqueeze(0)
    cp=torch.arange(start,start+seq.shape[1],dtype=torch.long)
    out=model(seq,past_key_values=cache,position_ids=pos,cache_position=cp,use_cache=False,return_dict=True)
    logits=out.logits.float()
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),t.reshape(-1))
    pred=logits.argmax(dim=-1)
    return loss, {"nll":loss.item(),"token_accuracy":(pred==t).float().mean().item(),
                  "exact":bool((pred==t).all().item())}

def selector_loss(model,selector,item):
    st,soft,pos=selector(item["contextual_old"])
    cache=build_selected_cache(item,st)
    loss,metrics=score_with_cache(model,item,cache)
    # Encourage different slots to represent different old locations, and sharper assignments.
    z=F.normalize(soft,p=2,dim=-1)
    gram=z@z.T
    off=(gram-torch.eye(gram.shape[0],device=gram.device)).pow(2).mean()
    entropy=-(soft.clamp_min(1e-8)*soft.clamp_min(1e-8).log()).sum(dim=-1).mean()
    objective=loss + 0.08*off + 0.003*entropy
    return objective,loss,off,entropy,metrics,pos

@torch.no_grad()
def baseline_metrics(model,item,kind):
    cache=build_full_cache(item) if kind=="full" else build_recent_cache(item)
    _,m=score_with_cache(model,item,cache)
    return m

@torch.no_grad()
def selected_metrics(model,selector,item):
    st,soft,pos=selector(item["contextual_old"])
    _,m=score_with_cache(model,item,build_selected_cache(item,st))
    m["selected_old_positions"]=pos.tolist()
    m["mean_entropy"]=(-(soft.clamp_min(1e-8)*soft.clamp_min(1e-8).log()).sum(dim=-1).mean()).item()
    return m

def main():
    started=time.time()
    print("loading",MODEL_ID,flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    selector=KVSelector(int(model.config.hidden_size))
    opt=torch.optim.AdamW(selector.parameters(),lr=LR,weight_decay=1e-4)

    # Every training table is asked about all four keys: memory must be query-independent.
    train_specs=[(i,q) for i in range(12) for q in range(4)]
    test_specs=[(20+i,q) for i in range(4) for q in range(4)]
    print("caching",len(train_specs),"train and",len(test_specs),"test cases",flush=True)
    train=[]
    for j,(i,q) in enumerate(train_specs):
        p,t,meta=make_case(i,q); x=cache_features(model,tok,p,t); x["meta"]=meta; train.append(x)
        if j%8==0: print(" cached train",j,meta,flush=True)
    test=[]
    for j,(i,q) in enumerate(test_specs):
        p,t,meta=make_case(i,q); x=cache_features(model,tok,p,t); x["meta"]=meta; test.append(x)
        if j%4==0: print(" cached test",j,meta,flush=True)

    curve=[]
    selector.train()
    order=list(range(len(train)))
    for step in range(TRAIN_STEPS):
        if step%len(train)==0: random.shuffle(order)
        item=train[order[step%len(train)]]
        opt.zero_grad(set_to_none=True)
        obj,loss,off,ent,metrics,pos=selector_loss(model,selector,item)
        obj.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(),1.0)
        opt.step()
        if step%8==0 or step==TRAIN_STEPS-1:
            rec={"step":step,"nll":loss.item(),"diversity_loss":off.item(),"entropy":ent.item(),
                 "temperature":selector.log_temp.exp().item(),"selected":pos.tolist()}
            curve.append(rec); print("step",rec,flush=True)

    selector.eval()
    evals=[]
    for item in test:
        fm=baseline_metrics(model,item,"full")
        rm=baseline_metrics(model,item,"recent")
        sm=selected_metrics(model,selector,item)
        evals.append({"meta":item["meta"],"prefix_tokens":int(item["prefix_ids"].shape[1]),
                      "old_tokens":int(item["old_len"]),"recent_tokens":RECENT,
                      "full":fm,"recent_only":rm,"learned_kv_selector":sm})
        print("eval",item["meta"],"full",fm["exact"],round(fm["nll"],3),
              "recent",rm["exact"],round(rm["nll"],3),
              "selected",sm["exact"],round(sm["nll"],3),sm["selected_old_positions"],flush=True)

    np,nt=natural_case(); natural=cache_features(model,tok,np,nt)
    natural_eval={"full":baseline_metrics(model,natural,"full"),
                  "recent_only":baseline_metrics(model,natural,"recent"),
                  "learned_kv_selector":selected_metrics(model,selector,natural)}

    def avg(path):
        vals=[]
        for e in evals:
            x=e
            for key in path: x=x[key]
            vals.append(float(x))
        return sum(vals)/len(vals)
    summary={
      "full_exact_rate":avg(["full","exact"]),
      "recent_exact_rate":avg(["recent_only","exact"]),
      "selected_exact_rate":avg(["learned_kv_selector","exact"]),
      "full_avg_nll":avg(["full","nll"]),
      "recent_avg_nll":avg(["recent_only","nll"]),
      "selected_avg_nll":avg(["learned_kv_selector","nll"]),
    }
    summary["selected_delta_nll_vs_full"]=summary["selected_avg_nll"]-summary["full_avg_nll"]
    summary["selected_improvement_vs_recent"]=summary["recent_avg_nll"]-summary["selected_avg_nll"]

    result={"model":MODEL_ID,"method":"straight-through learned selection of actual pre-RoPE-computed KV positions",
            "boundary_slots":SLOTS,"recent_tokens":RECENT,"max_prefix":MAX_PREFIX,
            "train_steps":TRAIN_STEPS,"rank":RANK,"selector_parameters":sum(p.numel() for p in selector.parameters()),
            "training_curve":curve,"heldout_retrieval":evals,"heldout_summary":summary,
            "natural_check":natural_eval,"elapsed_seconds":time.time()-started,
            "guardrails":["Bonsai weights are frozen.","Selector never sees the future query when selecting old positions; contextual_old is causal and precedes the recent question.",
                          "Forward pass retains exact K/V vectors from selected old positions; straight-through soft weights provide gradients.",
                          "Held-out tables use disjoint identifiers from training."]}
    OUT.write_text(json.dumps(result,indent=2))
    torch.save({"state_dict":selector.state_dict(),"config":{"hidden":int(model.config.hidden_size),"slots":SLOTS,"rank":RANK}},CKPT)
    print("summary",summary,flush=True)
    print("natural",natural_eval,flush=True)
    print("wrote",OUT,CKPT,flush=True)

if __name__=="__main__":
    main()
