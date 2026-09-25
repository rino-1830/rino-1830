import json, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ.get("MODEL_ID","prism-ml/Bonsai-1.7B-unpacked")
OUT=Path(os.environ.get("RESULT_PATH","boundary-compression-lab/results/segment_boundary.json"))
RECENT=int(os.environ.get("RECENT_TOKENS","48"))
OLD_LEN=int(os.environ.get("OLD_TOKENS","143"))
EPOCHS=int(os.environ.get("SEGMENT_EPOCHS","120"))
RANK=int(os.environ.get("SCORER_RANK","64"))
LR=float(os.environ.get("LR","0.003"))
BUDGETS=[16,24,32,48,64]
BLOCK=8
SEED=29
OUT.parent.mkdir(parents=True,exist_ok=True)
torch.manual_seed(SEED); random.seed(SEED); torch.set_num_threads(min(4,os.cpu_count() or 1))
LABELS=["ALPHA","BETA","GAMMA","DELTA"]
WORDS=["LANTERN","ORBIT","CEDAR","MICA","NOVA","RAVEN","EMBER","LOTUS","QUARTZ","MAPLE","SOLAR","VIOLET","COMET","FROST","IVORY","SABLE","CORAL","ONYX","AURORA","PINE","AMBER","FLINT","IRIS","LUNAR","BIRCH","OPAL","TERRA","ZEPHYR","CINDER","JADE","KITE"]

def code(i): return f"{WORDS[i%len(WORDS)]}-{4100+((i*379)%5800):04d}"
def legacy(c): return c.to_legacy_cache() if hasattr(c,"to_legacy_cache") else tuple(c)
def dyn(items):
    if hasattr(DynamicCache,"from_legacy_cache"): return DynamicCache.from_legacy_cache(tuple(items))
    c=DynamicCache()
    for i,(k,v) in enumerate(items): c.update(k,v,i)
    return c
def clone(items): return [(k.clone(),v.clone()) for k,v in items]

class Scorer(nn.Module):
    def __init__(self,h):
        super().__init__(); self.net=nn.Sequential(nn.Linear(h,RANK),nn.Tanh(),nn.Linear(RANK,1))
    def forward(self,x): return self.net(x.float()).squeeze(-1)

def filler_ids(tok,n,seed):
    texts=[
      "Routine maintenance notes mention weather, storage rooms, cables, roads, schedules, tools, and ordinary measurements. ",
      "The notebook contains unrelated inspections, replacement parts, operating procedures, materials, and daily checks. ",
      "Background observations describe benches, labels, transport, temperatures, doors, lighting, and equipment status. "]
    r=random.Random(seed); ids=[]
    while len(ids)<n+64:
        ids += tok(r.choice(texts),add_special_tokens=False).input_ids
    return ids[:n]

def build_table(tok,cid,insert_pos):
    cs=[code(cid*4+j) for j in range(4)]
    fact_text="".join(f"{LABELS[j]} = {cs[j]}\n" for j in range(4))
    facts=tok(fact_text,add_special_tokens=False).input_ids
    base=filler_ids(tok,OLD_LEN-len(facts),cid)
    pos=max(0,min(insert_pos,len(base)))
    old=base[:pos]+facts+base[pos:]
    old=old[:OLD_LEN]
    mask=torch.zeros(OLD_LEN)
    end=min(OLD_LEN,pos+len(facts)); mask[pos:end]=1
    packs=[]
    for q in range(4):
        question=f" Question: What is the exact identifier for {LABELS[q]}? Answer:"
        qids=tok(question,add_special_tokens=False).input_ids
        pad=filler_ids(tok,max(0,RECENT+1-len(qids)),1000+cid*10+q)
        prefix_tail=(pad+qids)[-(RECENT+1):]
        recent=prefix_tail[:-1]; last=prefix_tail[-1:]
        target=tok(" "+cs[q],add_special_tokens=False).input_ids[:12]
        packs.append({"q":q,"recent":torch.tensor([recent]),"last":torch.tensor([last]),"target":torch.tensor([target]),
                      "meta":{"case_id":cid,"query":LABELS[q],"answer":cs[q],"insert_pos":pos,"fact_tokens":len(facts)}})
    return torch.tensor([old]),mask.unsqueeze(0),packs

@torch.no_grad()
def prefill(model,ids):
    o=model(ids,use_cache=True,output_hidden_states=True,return_dict=True)
    items=[(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]
    h=o.hidden_states[-1].detach().float().clone()
    return items,h

@torch.no_grad()
def extend(model,olditems,recent):
    c=dyn(clone(olditems)); pos=torch.arange(OLD_LEN,OLD_LEN+recent.shape[1],dtype=torch.long)
    o=model(recent,past_key_values=c,position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    return [(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]

@torch.no_grad()
def score(model,pack,items):
    t=pack["target"]; seq=torch.cat([pack["last"],t[:,:-1]],1)
    st=OLD_LEN+RECENT; pos=torch.arange(st,st+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(clone(items)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    lg=o.logits.float(); loss=F.cross_entropy(lg.reshape(-1,lg.shape[-1]),t.reshape(-1)); pr=lg.argmax(-1)
    return {"nll":loss.item(),"token_accuracy":(pr==t).float().mean().item(),"exact":bool((pr==t).all().item())}

def select_positions(items,idx):
    return [(k.index_select(-2,idx).contiguous(),v.index_select(-2,idx).contiguous()) for k,v in items]

def token_topk(logits,b):
    return torch.topk(logits[0],min(b,logits.shape[-1])).indices.sort().values

def block_topk(logits,b):
    n=logits.shape[-1]; scores=[]; spans=[]
    for s in range(0,n,BLOCK):
        e=min(n,s+BLOCK); scores.append(logits[0,s:e].mean()); spans.append((s,e))
    order=torch.argsort(torch.stack(scores),descending=True)
    keep=[]; cap=0
    for j in order.tolist():
        s,e=spans[j]; seg=list(range(s,e))
        if cap+len(seg)>b: seg=seg[:max(0,b-cap)]
        keep+=seg; cap+=len(seg)
        if cap>=b: break
    return torch.tensor(sorted(keep),dtype=torch.long)

def span_oracle(mask,b):
    fact=torch.where(mask[0]>0.5)[0].tolist()
    if len(fact)>=b: fact=fact[:b]
    else:
        used=set(fact); extra=[i for i in range(OLD_LEN) if i not in used][:b-len(fact)]; fact+=extra
    return torch.tensor(sorted(fact),dtype=torch.long)

def train(model,tok,scorer):
    data=[]
    positions=[0,16,32,48,64,80,96]
    for cid in range(18):
        pos=positions[cid%len(positions)]
        old,mask,_=build_table(tok,cid,pos); _,h=prefill(model,old); data.append((h,mask))
    opt=torch.optim.AdamW(scorer.parameters(),lr=LR,weight_decay=1e-4)
    curve=[]
    for ep in range(EPOCHS):
        order=list(range(len(data))); random.Random(SEED+ep).shuffle(order); total=0
        for i in order:
            h,m=data[i]; lg=scorer(h)
            posw=torch.tensor((OLD_LEN-m.sum().item())/max(1.0,m.sum().item()))
            loss=F.binary_cross_entropy_with_logits(lg,m,pos_weight=posw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=loss.item()
        if ep%20==0 or ep==EPOCHS-1:
            rec={"epoch":ep,"loss":total/len(data)}; curve.append(rec); print("train",rec,flush=True)
    return curve

@torch.no_grad()
def evaluate(model,tok,scorer):
    rows=[]; test_positions=[8,24,40,56,72,88]
    for z,cid in enumerate(range(30,36)):
        old,mask,packs=build_table(tok,cid,test_positions[z])
        olditems,h=prefill(model,old); logits=scorer(h)
        for pack in packs:
            full=score(model,pack,extend(model,olditems,pack["recent"]))
            recentitems=[(k[...,OLD_LEN:,:],v[...,OLD_LEN:,:]) for k,v in extend(model,olditems,pack["recent"])]
            recent=score(model,pack,recentitems)
            budgets={}
            for b in BUDGETS:
                methods={}
                for name,idx in [
                    ("token",token_topk(logits,b)),
                    ("block",block_topk(logits,b)),
                    ("span_oracle",span_oracle(mask,b)),
                ]:
                    kept=select_positions(olditems,idx)
                    streamed=extend(model,kept,pack["recent"])
                    methods[name]={"metrics":score(model,pack,streamed),"selected":idx.tolist()}
                budgets[str(b)]=methods
            rows.append({"meta":pack["meta"],"full":full,"recent_only":recent,"budgets":budgets})
        print("eval table",cid,"pos",test_positions[z],flush=True)
    return rows

def summary(rows):
    out={"full":{"avg_nll":sum(r["full"]["nll"] for r in rows)/len(rows),"exact_rate":sum(r["full"]["exact"] for r in rows)/len(rows)},
         "recent_only":{"avg_nll":sum(r["recent_only"]["nll"] for r in rows)/len(rows),"exact_rate":sum(r["recent_only"]["exact"] for r in rows)/len(rows)}}
    for b in BUDGETS:
        out[str(b)]={}
        for m in ["token","block","span_oracle"]:
            xs=[r["budgets"][str(b)][m]["metrics"] for r in rows]
            out[str(b)][m]={"avg_nll":sum(x["nll"] for x in xs)/len(xs),"exact_rate":sum(x["exact"] for x in xs)/len(xs)}
    return out

def main():
    st=time.time(); print("loading",MODEL_ID,flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(MODEL_ID); scorer=Scorer(int(model.config.hidden_size))
    curve=train(model,tok,scorer); scorer.eval(); rows=evaluate(model,tok,scorer); sm=summary(rows)
    result={"model":MODEL_ID,"method":"random-position fact spans: causal token scorer vs contiguous block selection vs span oracle",
            "old_tokens":OLD_LEN,"recent_tokens":RECENT,"budgets":BUDGETS,"block_size":BLOCK,
            "training_curve":curve,"heldout":rows,"summary":sm,"elapsed_seconds":time.time()-st,
            "guardrails":["Future question is absent when old-prefix boundary selection is computed.","Fact table location is varied in training and held-out at unseen offsets.","span_oracle uses ground-truth fact positions only as a capacity upper bound.","block selection uses the same learned causal token logits but preserves contiguous regions."]}
    OUT.write_text(json.dumps(result,indent=2)); print("summary",json.dumps(sm),flush=True); print("wrote",OUT,flush=True)

if __name__=="__main__": main()
