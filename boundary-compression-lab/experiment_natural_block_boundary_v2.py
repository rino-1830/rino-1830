import gc, json, math, os, random, time
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
TRAIN_N=int(os.environ.get("TRAIN_WINDOWS","10")); EVAL_N=int(os.environ.get("EVAL_WINDOWS","16"))
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

class Gate(nn.Module):
    def __init__(self,n=6): super().__init__(); self.lin=nn.Linear(n,1)
    def forward(self,x): return self.lin(x).squeeze(-1)

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
    for j in bi.tolist(): pos+=list(range(j*BLOCK,min(OLD,j*BLOCK+BLOCK)))
    idx=torch.tensor(sorted(set(pos))[:budget],dtype=torch.long)
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
        out["kl_to_full"]=float(F.kl_div(F.log_softmax(logits,dim=-1),F.softmax(full_logits,dim=-1),reduction="batchmean"))
    return out

def recent_only(items):
    return [(k[...,-1:,:].clone(),v[...,-1:,:].clone()) for k,v in items]

def make_chunks(tok,split,n,offset):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split)
    docs=[]; cur=[]
    for row in ds:
        t=row["text"]; stripped=t.strip()
        if stripped.startswith("=") and stripped.endswith("=") and cur:
            docs.append("\n".join(cur)); cur=[t]
        else: cur.append(t)
    if cur: docs.append("\n".join(cur))
    need=OLD+RECENT+TARGET; candidates=[]
    for d in docs:
        ids=tok(d,add_special_tokens=False,return_tensors="pt").input_ids[0]
        if len(ids)>=need+8: candidates.append(ids)
    rng=random.Random(SEED+offset+(0 if split=="train" else 999)); order=list(range(len(candidates))); rng.shuffle(order)
    chunks=[]
    while len(chunks)<n:
        ids=candidates[order[len(chunks)%len(order)]]; maxs=len(ids)-need
        st=rng.randrange(0,max(1,maxs+1)); z=ids[st:st+need].unsqueeze(0)
        chunks.append((z[:,:OLD],z[:,OLD:OLD+RECENT],z[:,OLD+RECENT:]))
    return chunks

def gate_features(score):
    x=score[0].float(); p=F.softmax(x,dim=-1)
    top=torch.topk(x,min(4,x.numel())).values
    entropy=-(p*p.clamp_min(1e-12).log()).sum()
    margin=top[0]-top[1] if top.numel()>1 else top[0]*0
    return torch.stack([x.max(),x.mean(),x.std(unbiased=False),top.mean(),margin,entropy])

def train_scorer(model,train,scorer):
    data=[]
    for i,(old,recent,target) in enumerate(train):
        items,h=prefill(model,old); mass,nll=grad_teacher(model,items,recent,target)
        data.append((block_h(h),mass.unsqueeze(0))); print("teacher",i,round(nll,4),flush=True)
        del items,h; gc.collect()
    opt=torch.optim.AdamW(scorer.parameters(),lr=LR,weight_decay=1e-4)
    for ep in range(EPOCHS):
        total=0
        for h,mass in data:
            lg=scorer(h); loss=F.kl_div(F.log_softmax(lg,-1),mass,reduction="batchmean")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=float(loss)
        if ep%20==0 or ep==EPOCHS-1: print("epoch",ep,total/len(data),flush=True)

def train_gate(model,train,scorer,budget=64):
    feats=[]; labels=[]; margins=[]
    scorer.eval()
    for old,recent,target in train:
        items,h=prefill(model,old); score=scorer(block_h(h)).detach()
        sel,_=select(items,score,budget)
        rn=metrics(logits_for(model,recent_only(items),recent,target),target)["nll"]
        bn=metrics(logits_for(model,sel,recent,target),target)["nll"]
        feats.append(gate_features(score)); labels.append(float(bn<rn)); margins.append(rn-bn)
    X=torch.stack(feats); y=torch.tensor(labels)
    mu=X.mean(0); sd=X.std(0,unbiased=False).clamp_min(1e-5); Xn=(X-mu)/sd
    gate=Gate(X.shape[1]); opt=torch.optim.AdamW(gate.parameters(),lr=0.05,weight_decay=1e-3)
    for _ in range(300):
        lg=gate(Xn); loss=F.binary_cross_entropy_with_logits(lg,y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print("gate labels",labels,"margins",[round(x,3) for x in margins],flush=True)
    return gate.eval(),mu,sd

@torch.no_grad()
def evaluate(model,evals,scorer,gate,mu,sd):
    rows=[]
    for i,(old,recent,target) in enumerate(evals):
        items,h=prefill(model,old); score=scorer(block_h(h)).detach()
        full_logits=logits_for(model,items,recent,target); full=metrics(full_logits,target)
        recent_logits=logits_for(model,recent_only(items),recent,target); recent_m=metrics(recent_logits,target,full_logits)
        budgets={}
        for b in BUDGETS:
            sel,pos=select(items,score,b); lg=logits_for(model,sel,recent,target)
            budgets[str(b)]={**metrics(lg,target,full_logits),"selected":pos.tolist()}
        feat=gate_features(score); prob=float(torch.sigmoid(gate(((feat-mu)/sd).unsqueeze(0)))[0])
        choose_boundary=prob>=0.5
        chosen=budgets["64"]["nll"] if choose_boundary else recent_m["nll"]
        oracle=min(recent_m["nll"],budgets["64"]["nll"])
        rows.append({"i":i,"full":full,"recent":recent_m,"budgets":budgets,
                     "gate_prob_boundary":prob,"gate_choice":"boundary64" if choose_boundary else "recent",
                     "gate_nll":chosen,"oracle_gate_nll":oracle,
                     "oracle_choice":"boundary64" if budgets["64"]["nll"]<recent_m["nll"] else "recent"})
        print("eval",i,round(full["nll"],3),round(recent_m["nll"],3),round(budgets["64"]["nll"],3),round(prob,3),flush=True)
    return rows

def summarize(rows):
    mean=lambda fn: sum(fn(r) for r in rows)/len(rows)
    out={"full_nll":mean(lambda r:r["full"]["nll"]),"recent_nll":mean(lambda r:r["recent"]["nll"]),
         "gate_nll":mean(lambda r:r["gate_nll"]),"oracle_gate_nll":mean(lambda r:r["oracle_gate_nll"]),
         "gate_accuracy_vs_oracle_choice":mean(lambda r:float(r["gate_choice"]==r["oracle_choice"])),
         "boundary64_win_rate":mean(lambda r:float(r["budgets"]["64"]["nll"]<r["recent"]["nll"]))}
    for b in BUDGETS: out[str(b)]={"nll":mean(lambda r,b=b:r["budgets"][str(b)]["nll"]),
                                     "kl_to_full":mean(lambda r,b=b:r["budgets"][str(b)]["kl_to_full"])}
    return out

def main():
    st=time.time(); print("loading",MODEL_ID,flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    train=make_chunks(tok,"train",TRAIN_N,0); ev=make_chunks(tok,"validation",EVAL_N,1)
    scorer=BlockScorer(int(model.config.hidden_size)); train_scorer(model,train,scorer); scorer.eval()
    gate,mu,sd=train_gate(model,train,scorer,64); rows=evaluate(model,ev,scorer,gate,mu,sd); sm=summarize(rows)
    result={"model":MODEL_ID,"dataset":"Salesforce/wikitext wikitext-2-raw-v1","old":OLD,"recent":RECENT,"target":TARGET,
            "train_windows":TRAIN_N,"eval_windows":EVAL_N,"summary":sm,"heldout":rows,"elapsed_seconds":time.time()-st,
            "guardrails":["Train/eval use disjoint WikiText splits and stay within reconstructed document boundaries.",
                          "Boundary scorer and gate see only old-prefix representations at inference.",
                          "Future targets are used only to create training labels and held-out evaluation metrics."]}
    OUT.write_text(json.dumps(result,indent=2)); print("summary",json.dumps(sm),flush=True)

if __name__=="__main__": main()
