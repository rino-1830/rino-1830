import gc, json, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ.get("MODEL_ID","prism-ml/Bonsai-1.7B-unpacked")
OUT=Path(os.environ.get("RESULT_PATH","boundary-compression-lab/results/natural_block_boundary.json"))
OLD=int(os.environ.get("OLD_TOKENS","192")); RECENT=int(os.environ.get("RECENT_TOKENS","32")); TARGET=int(os.environ.get("TARGET_TOKENS","8"))
BLOCK=8; RANK=64; EPOCHS=80; LR=0.003; SEED=73; BUDGETS=[32,64,96]
TRAIN_N=8; EVAL_N=8
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
    text="\n".join(x["text"] for x in ds if x["text"].strip())
    ids=tok(text,add_special_tokens=False,return_tensors="pt").input_ids[0]
    need=OLD+RECENT+TARGET; chunks=[]
    stride=max(need+17,257)
    for i in range(n):
        s=offset+i*stride
        if s+need>len(ids): s=(i*stride)%(len(ids)-need-1)
        z=ids[s:s+need].unsqueeze(0)
        chunks.append((z[:,:OLD],z[:,OLD:OLD+RECENT],z[:,OLD+RECENT:]))
    return chunks

def train(model,tok,scorer):
    chunks=make_chunks(tok,"train",TRAIN_N,512); data=[]
    for i,(old,recent,target) in enumerate(chunks):
        items,h=prefill(model,old); mass,loss=grad_teacher(model,items,recent,target); data.append((block_h(h),mass.unsqueeze(0)))
        print("teacher",i,"nll",round(loss,4),flush=True); del items,h; gc.collect()
    opt=torch.optim.AdamW(scorer.parameters(),lr=LR,weight_decay=1e-4)
    for ep in range(EPOCHS):
        order=list(range(len(data))); random.Random(SEED+ep).shuffle(order)
        for i in order:
            h,m=data[i]; lg=scorer(h); loss=F.kl_div(F.log_softmax(lg,-1),m,reduction="batchmean")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep%20==0: print("epoch",ep,flush=True)

def main():
    st=time.time(); print("loading model",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID); model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    scorer=BlockScorer(int(model.config.hidden_size)); train(model,tok,scorer); scorer.eval()
    chunks=make_chunks(tok,"validation",EVAL_N,1024); rows=[]
    for i,(old,recent,target) in enumerate(chunks):
        items,h=prefill(model,old); pred=scorer(block_h(h)).detach(); teacher,_=grad_teacher(model,items,recent,target); teacher=teacher.unsqueeze(0)
        full_lg=logits_for(model,items,recent,target); row={"full":metrics(full_lg,target),"budgets":{}}
        # Recent-only uses no old KV but preserves absolute positions.
        empty=[(k[...,:0,:],v[...,:0,:]) for k,v in items]; recent_lg=logits_for(model,empty,recent,target)
        row["recent_only"]=metrics(recent_lg,target,full_lg)
        for b in BUDGETS:
            ps,_=select(items,pred,b); ts,_=select(items,teacher,b)
            pl=logits_for(model,ps,recent,target); tl=logits_for(model,ts,recent,target)
            row["budgets"][str(b)]={"learned":metrics(pl,target,full_lg),"teacher":metrics(tl,target,full_lg)}
        rows.append(row); print("eval",i,flush=True)
    def avg(path):
        vs=[]
        for r in rows:
            x=r
            for k in path:x=x[k]
            vs.append(float(x))
        return sum(vs)/len(vs)
    sm={"full":{"nll":avg(["full","nll"])},"recent_only":{"nll":avg(["recent_only","nll"]),"kl":avg(["recent_only","kl_to_full"])}}
    for b in BUDGETS:
        sm[str(b)]={}
        for m in ["learned","teacher"]:
            sm[str(b)][m]={"nll":avg(["budgets",str(b),m,"nll"]),"kl":avg(["budgets",str(b),m,"kl_to_full"])}
    OUT.write_text(json.dumps({"summary":sm,"rows":rows,"elapsed":time.time()-st},indent=2)); print(json.dumps(sm,indent=2),flush=True)
if __name__=="__main__": main()
