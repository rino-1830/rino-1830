import gc, json, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ.get("MODEL_ID","prism-ml/Bonsai-1.7B-unpacked")
OUT=Path(os.environ.get("RESULT_PATH","boundary-compression-lab/results/natural_block_boundary_v2.json"))
OLD=int(os.environ.get("OLD_TOKENS","192")); RECENT=int(os.environ.get("RECENT_TOKENS","32")); TARGET=int(os.environ.get("TARGET_TOKENS","16"))
BLOCK=8; RANK=64; EPOCHS=80; LR=0.003; SEED=73; BUDGETS=[32,64,96]
TRAIN_N=10; EVAL_N=16
OUT.parent.mkdir(parents=True,exist_ok=True); torch.manual_seed(SEED); random.seed(SEED); torch.set_num_threads(min(4,os.cpu_count() or 1))

def legacy(c): return c.to_legacy_cache() if hasattr(c,"to_legacy_cache") else tuple(c)
def dyn(items):
    c=DynamicCache()
    for i,(k,v) in enumerate(items): c.update(k,v,i)
    return c
def clone(items): return [(k.clone(),v.clone()) for k,v in items]

class BlockScorer(nn.Module):
    def __init__(self,h):
        super().__init__(); self.net=nn.Sequential(nn.Linear(h,RANK),nn.Tanh(),nn.Linear(RANK,1))
    def forward(self,x): return self.net(x.float()).squeeze(-1)

def block_h(h): return torch.stack([h[:,s:min(h.shape[1],s+BLOCK),:].mean(1) for s in range(0,h.shape[1],BLOCK)],1)
def block_mass(m):
    z=torch.stack([m[s:min(m.numel(),s+BLOCK)].sum() for s in range(0,m.numel(),BLOCK)])
    return z/z.sum().clamp_min(1e-12)

@torch.no_grad()
def prefill(model,old):
    o=model(old,use_cache=True,output_hidden_states=True,return_dict=True)
    return [(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)],o.hidden_states[-1].detach().float().clone()

def grad_teacher(model,items,recent,target):
    leaves=[]; cache=[]
    for k,v in items:
        kk=k.detach().clone().requires_grad_(True); vv=v.detach().clone().requires_grad_(True)
        leaves.append((kk,vv)); cache.append((kk,vv))
    seq=torch.cat([recent,target[:,:-1]],1); r=recent.shape[1]; t=target.shape[1]
    pos=torch.arange(OLD,OLD+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(cache),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    lg=o.logits[:,r-1:r-1+t,:].float()
    loss=F.cross_entropy(lg.reshape(-1,lg.shape[-1]),target.reshape(-1)); loss.backward()
    ls=[]
    for k,v in leaves:
        ks=(k.grad.float()*k.detach().float()).sum(-1).abs().mean((0,1))
        vs=(v.grad.float()*v.detach().float()).sum(-1).abs().mean((0,1))
        z=ks+vs; ls.append(z/z.sum().clamp_min(1e-12))
    imp=torch.stack(ls).mean(0); imp=(imp/imp.sum().clamp_min(1e-12)+1e-12).sqrt(); imp=imp/imp.sum()
    val=float(loss.detach()); del o,cache,leaves; gc.collect()
    return block_mass(imp).detach(),val

def select(items,score,budget):
    n=max(1,(budget+BLOCK-1)//BLOCK); bi=torch.topk(score[0],min(n,score.shape[-1])).indices
    pos=[]
    for j in bi.tolist():
        pos+=list(range(j*BLOCK,min(OLD,j*BLOCK+BLOCK)))
        if len(pos)>=budget: break
    idx=torch.tensor(sorted(pos[:budget]),dtype=torch.long)
    return [(k.index_select(-2,idx).contiguous(),v.index_select(-2,idx).contiguous()) for k,v in items],idx

@torch.no_grad()
def logits_for(model,items,recent,target):
    pos=torch.arange(OLD,OLD+recent.shape[1],dtype=torch.long)
    o=model(recent,past_key_values=dyn(clone(items)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    first=o.logits[:,-1:,:].float(); c=[(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]
    if target.shape[1]>1:
        inp=target[:,:-1]; p2=torch.arange(OLD+recent.shape[1],OLD+recent.shape[1]+inp.shape[1],dtype=torch.long)
        q=model(inp,past_key_values=dyn(c),position_ids=p2.unsqueeze(0),cache_position=p2,use_cache=False,return_dict=True)
        lg=torch.cat([first,q.logits.float()],1)
    else: lg=first
    return lg[:,:target.shape[1],:]

def metrics(logits,target,full_logits=None):
    ce=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1)))
    out={"nll":ce}
    if full_logits is not None:
        kl=F.kl_div(F.log_softmax(logits,dim=-1),F.softmax(full_logits,dim=-1),reduction="batchmean")
        out["kl_to_full"]=float(kl)
    return out

def make_chunks(tok,split,n,offset):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split)
    # Reconstruct article-like documents from headings so windows never cross document boundaries.
    docs=[]; cur=[]
    for row in ds:
        t=row["text"]
        stripped=t.strip()
        if stripped.startswith("=") and stripped.endswith("=") and cur:
            docs.append("\n".join(cur)); cur=[t]
        else:
            cur.append(t)
    if cur: docs.append("\n".join(cur))
    need=OLD+RECENT+TARGET
    candidates=[]
    for d in docs:
        ids=tok(d,add_special_tokens=False,return_tensors="pt").input_ids[0]
        if len(ids)>=need+8: candidates.append(ids)
    if not candidates:
        raise RuntimeError("no sufficiently long within-document sequences")
    chunks=[]
    rng=random.Random(SEED+offset+(0 if split=="train" else 999))
    order=list(range(len(candidates))); rng.shuffle(order)
    while len(chunks)<n:
        ids=candidates[order[len(chunks)%len(order)]]
        maxs=len(ids)-need
        st=rng.randrange(0,max(1,maxs+1))
        z=ids[st:st+need].unsqueeze(0)
        chunks.append((z[:,:OLD],z[:,OLD:OLD+RECENT],z[:,OLD+RECENT:]))
    return chunks

