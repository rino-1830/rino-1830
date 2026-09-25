import ctypes, gc, json, math, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ.get("MODEL_ID","prism-ml/Bonsai-1.7B-unpacked")
MODEL_DTYPE_NAME=os.environ.get("MODEL_DTYPE","float32")
MODEL_DTYPE={"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}[MODEL_DTYPE_NAME]
OUT=Path(os.environ.get("RESULT_PATH","boundary-compression-lab/results/predictive_scorer.json"))
CKPT=Path(os.environ.get("CKPT_PATH","boundary-compression-lab/results/predictive_scorer.pt"))
RECENT=int(os.environ.get("RECENT_TOKENS","48"))
MAX_PREFIX=int(os.environ.get("MAX_PREFIX","192"))
EPOCHS=int(os.environ.get("SCORER_EPOCHS","160"))
RANK=int(os.environ.get("SCORER_RANK","64"))
LR=float(os.environ.get("LR","0.003"))
THREADS=int(os.environ.get("TORCH_THREADS","4"))
BUDGETS=[int(x) for x in os.environ.get("BUDGETS","8,16,24,32,48,64").split(",")]
SEED=19
OUT.parent.mkdir(parents=True,exist_ok=True)
torch.manual_seed(SEED); random.seed(SEED); torch.set_num_threads(min(THREADS,os.cpu_count() or 1))
LABELS=["ALPHA","BETA","GAMMA","DELTA"]
WORDS=["LANTERN","ORBIT","CEDAR","MICA","NOVA","RAVEN","EMBER","LOTUS","QUARTZ","MAPLE","SOLAR","VIOLET","COMET","FROST","IVORY","DELTA","SABLE","CORAL","ONYX","AURORA","PINE","AMBER","FLINT","IRIS","LUNAR","BIRCH","OPAL","TERRA","ZEPHYR","CINDER","JADE","KITE"]

def code(i): return f"{WORDS[i%len(WORDS)]}-{4100+((i*379)%5800):04d}"
def filler(seed,repeats=10):
    v=["A field notebook records routine observations about tools, roads, materials, weather, and schedules. ",
       "The maintenance log lists ordinary checks, measurements, replacement parts, and inspection notes. ",
       "Unrelated notes describe storage rooms, work benches, cables, labels, and daily operating procedures. ",
       "The report contains mundane details that should not replace the identifiers stored in the table. "]
    r=random.Random(seed); return "".join(r.choice(v) for _ in range(repeats))
def case(case_id,q):
    cs=[code(case_id*4+j) for j in range(4)]
    table="\n".join(f"{LABELS[j]} = {cs[j]}" for j in range(4))
    p="Memorize the following identifier table exactly.\n"+table+"\n\n"+filler(case_id)+f"\nQuestion: What is the exact identifier for {LABELS[q]}?\nAnswer:"
    return p," "+cs[q],{"case_id":case_id,"query":LABELS[q],"answer":cs[q]}
def natural_case():
    return ("A transformer language model predicts the next token from a sequence of earlier tokens. During autoregressive inference, attention keys and values from earlier positions are cached so they do not need to be recomputed. As the context grows, this cache can become a large fraction of total memory use. One possible strategy is to preserve only the information from the distant prefix that is useful for predicting the continuation. In that setting, the main question is whether a small boundary state can replace most of the old cache without changing ","the model prediction very much.")
def encode(tok,p,t):
    p=tok(p,add_special_tokens=False,return_tensors="pt").input_ids
    t=tok(t,add_special_tokens=False,return_tensors="pt").input_ids[:,:12]
    if p.shape[1]>MAX_PREFIX:
        p=torch.cat([p[:,:MAX_PREFIX-RECENT],p[:,-RECENT:]],1)
    return p,t
def legacy(c): return c.to_legacy_cache() if hasattr(c,"to_legacy_cache") else tuple(c)
def dyn(items):
    if hasattr(DynamicCache,"from_legacy_cache"): return DynamicCache.from_legacy_cache(tuple(items))
    c=DynamicCache()
    for i,(k,v) in enumerate(items): c.update(k,v,i)
    return c
def clone(items): return [(k.clone(),v.clone()) for k,v in items]

class Scorer(nn.Module):
    def __init__(self,h,rank=RANK):
        super().__init__(); self.net=nn.Sequential(nn.Linear(h,rank),nn.Tanh(),nn.Linear(rank,1))
    def forward(self,x): return self.net(x.float()).squeeze(-1)

def prep(tok,case_id):
    packs=[]; old=None
    for q in range(4):
        p,t,m=case(case_id,q); p,t=encode(tok,p,t); cached=p[:,:-1]; ol=max(1,cached.shape[1]-RECENT)
        oi=cached[:,:ol]; ri=cached[:,ol:]
        if old is None: old=oi
        elif not torch.equal(old,oi): raise RuntimeError("old prefix differs across future queries")
        packs.append({"p":p,"t":t,"recent":ri,"last":p[:,-1:],"old_len":ol,"cached_len":cached.shape[1],"meta":m})
    return old,packs

@torch.no_grad()
def prefill(model,ids,attn=False):
    o=model(ids,use_cache=True,output_hidden_states=True,output_attentions=attn,return_dict=True)
    items=[(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]
    h=o.hidden_states[-1].detach().float().clone()
    return items,h,o.attentions

@torch.no_grad()
def trim_memory():
    gc.collect()
    try:
        libc=ctypes.CDLL(None)
        if hasattr(libc,"malloc_trim"): libc.malloc_trim(0)
    except Exception:
        pass

def future_saliency(model,p):
    # Teacher only: future/recent queries -> old positions, aggregated across layers/heads.
    o=model(p,use_cache=False,output_attentions=True,return_dict=True)
    ol=max(1,p.shape[1]-1-RECENT)
    scores=[]
    for a in o.attentions:
        x=a[0,:,ol:,:ol].float()
        # Any future token/head may expose a sparse dependency; max over q, mean over heads.
        scores.append(x.amax(dim=1).mean(dim=0))
    s=torch.stack(scores).mean(dim=0)
    return s.clamp_min(0)

@torch.no_grad()
def teacher_for_table(model,tok,case_id):
    old,packs=prep(tok,case_id)
    agg=None
    for pack in packs:
        s=future_saliency(model,pack["p"])
        agg=s if agg is None else agg+s
    agg=agg/len(packs)
    # Mix normalized attention mass with a binary top-region target for ranking stability.
    mass=(agg+1e-8).sqrt(); mass=mass/mass.sum()
    k=min(48,mass.numel()); pos=torch.topk(mass,k).indices
    binary=torch.zeros_like(mass); binary[pos]=1
    return old,packs,mass,binary

@torch.no_grad()
def old_features(model,old):
    items,h,_=prefill(model,old)
    return items,h

def train_scorer(model,tok,scorer):
    dataset=[]
    print("building future-influence teacher",flush=True)
    for cid in range(12):
        old,packs,mass,binary=teacher_for_table(model,tok,cid)
        _,h=old_features(model,old)
        dataset.append((h,mass.unsqueeze(0),binary.unsqueeze(0)))
        print(" teacher",cid,"top",torch.topk(mass,min(12,mass.numel())).indices.tolist(),flush=True)
    opt=torch.optim.AdamW(scorer.parameters(),lr=LR,weight_decay=1e-4)
    pw=torch.tensor(2.0)
    curve=[]
    for ep in range(EPOCHS):
        order=list(range(len(dataset))); random.Random(SEED+ep).shuffle(order)
        total=0
        for i in order:
            h,mass,binary=dataset[i]; logits=scorer(h)
            bce=F.binary_cross_entropy_with_logits(logits,binary,pos_weight=pw)
            logp=F.log_softmax(logits,dim=-1)
            kl=F.kl_div(logp,mass,reduction="batchmean")
            loss=bce+0.35*kl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=loss.item()
        if ep%20==0 or ep==EPOCHS-1:
            rec={"epoch":ep,"loss":total/len(dataset)}; curve.append(rec); print("score-train",rec,flush=True)
    return curve

@torch.no_grad()
def extend(model,items,recent,start):
    c=dyn(clone(items)); n=recent.shape[1]; pos=torch.arange(start,start+n,dtype=torch.long)
    o=model(recent,past_key_values=c,position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    return [(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]
def choose(items,scores,k):
    k=min(k,scores.numel()); idx=torch.topk(scores[0],k).indices.sort().values
    return [(x.index_select(-2,idx).contiguous(),v.index_select(-2,idx).contiguous()) for x,v in items],idx
@torch.no_grad()
def score(model,pack,items):
    t=pack["t"]; seq=torch.cat([pack["last"],t[:,:-1]],1); st=pack["cached_len"]
    pos=torch.arange(st,st+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(clone(items)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    lg=o.logits.float(); loss=F.cross_entropy(lg.reshape(-1,lg.shape[-1]),t.reshape(-1)); pr=lg.argmax(-1)
    return {"nll":loss.item(),"token_accuracy":(pr==t).float().mean().item(),"exact":bool((pr==t).all().item())}

@torch.no_grad()
def eval_table(model,tok,scorer,cid,with_teacher=True):
    old,packs=prep(tok,cid); olditems,h=old_features(model,old); pred=scorer(h)
    teacher=None
    if with_teacher:
        _,_,mass,_=teacher_for_table(model,tok,cid); teacher=mass.unsqueeze(0)
    rows=[]
    for pack in packs:
        full=extend(model,olditems,pack["recent"],pack["old_len"]); fm=score(model,pack,full)
        recent=[(k[...,pack["old_len"]:,:],v[...,pack["old_len"]:,:]) for k,v in full]; rm=score(model,pack,recent)
        methods={}
        for b in BUDGETS:
            sel,idx=choose(olditems,pred,b); streamed=extend(model,sel,pack["recent"],pack["old_len"]); methods[str(b)]={"metrics":score(model,pack,streamed),"selected":idx.tolist()}
            if teacher is not None:
                ts,ti=choose(olditems,teacher,b); tm=score(model,pack,extend(model,ts,pack["recent"],pack["old_len"]))
                methods[str(b)]["teacher_oracle"]=tm; methods[str(b)]["teacher_selected"]=ti.tolist()
        rows.append({"meta":pack["meta"],"full":fm,"recent_only":rm,"budgets":methods})
    return rows

def summarize(rows):
    out={}
    for b in BUDGETS:
        xs=[r["budgets"][str(b)]["metrics"] for r in rows]
        ts=[r["budgets"][str(b)].get("teacher_oracle") for r in rows]
        out[str(b)]={"avg_nll":sum(x["nll"] for x in xs)/len(xs),"exact_rate":sum(x["exact"] for x in xs)/len(xs)}
        if all(t is not None for t in ts):
            out[str(b)]["teacher_avg_nll"]=sum(t["nll"] for t in ts)/len(ts); out[str(b)]["teacher_exact_rate"]=sum(t["exact"] for t in ts)/len(ts)
    out["full"]={"avg_nll":sum(r["full"]["nll"] for r in rows)/len(rows),"exact_rate":sum(r["full"]["exact"] for r in rows)/len(rows)}
    out["recent_only"]={"avg_nll":sum(r["recent_only"]["nll"] for r in rows)/len(rows),"exact_rate":sum(r["recent_only"]["exact"] for r in rows)/len(rows)}
    return out

def main():
    start=time.time(); print("loading",MODEL_ID,flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=MODEL_DTYPE,low_cpu_mem_usage=True,attn_implementation="eager")
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(MODEL_ID); scorer=Scorer(int(model.config.hidden_size))
    curve=train_scorer(model,tok,scorer); scorer.eval()
    rows=[]
    for cid in range(20,24):
        rr=eval_table(model,tok,scorer,cid,True); rows.extend(rr)
        print("eval-table",cid,flush=True)
    summary=summarize(rows); print("summary",json.dumps(summary),flush=True)
    result={"model":MODEL_ID,"method":"causal scalar predictive-influence scorer trained from future-attention teacher","budgets":BUDGETS,"recent_tokens":RECENT,"max_prefix":MAX_PREFIX,"scorer_params":sum(p.numel() for p in scorer.parameters()),"training_curve":curve,"heldout":rows,"summary":summary,"elapsed_seconds":time.time()-start,"guardrails":["Future queries are used only to construct teacher saliency during training.","The learned scorer sees only causal old-prefix hidden states at inference.","Top-k sets are nested across budgets because one scalar score is learned per old token.","Teacher-oracle metrics test whether the attention-derived supervision itself contains enough information."]}
    OUT.write_text(json.dumps(result,indent=2)); torch.save({"state_dict":scorer.state_dict(),"rank":RANK},CKPT); print("wrote",OUT,flush=True)
if __name__=="__main__": main()
