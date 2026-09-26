import gc, json, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ.get("MODEL_ID","prism-ml/Bonsai-1.7B-unpacked")
OUT=Path(os.environ.get("RESULT_PATH","boundary-compression-lab/results/realtext_gradient_block.json"))
OLD=int(os.environ.get("OLD_TOKENS","143")); RECENT=int(os.environ.get("RECENT_TOKENS","48")); TARGET=int(os.environ.get("TARGET_TOKENS","8"))
BLOCK=8; RANK=64; TRAIN_N=int(os.environ.get("TRAIN_WINDOWS","6")); TEST_N=int(os.environ.get("TEST_WINDOWS","6")); EPOCHS=100
BUDGETS=[32,64]; SEED=73
OUT.parent.mkdir(parents=True,exist_ok=True)
torch.manual_seed(SEED); random.seed(SEED); torch.set_num_threads(min(4,os.cpu_count() or 1))

def legacy(c): return c.to_legacy_cache() if hasattr(c,"to_legacy_cache") else tuple(c)
def dyn(items):
    c=DynamicCache()
    for i,(k,v) in enumerate(items): c.update(k,v,i)
    return c
def clone(items): return [(k.clone(),v.clone()) for k,v in items]

class Scorer(nn.Module):
    def __init__(self,h):
        super().__init__(); self.net=nn.Sequential(nn.Linear(h,RANK),nn.Tanh(),nn.Linear(RANK,1))
    def forward(self,x): return self.net(x.float()).squeeze(-1)

def block_h(h):
    return torch.stack([h[:,s:min(h.shape[1],s+BLOCK),:].mean(1) for s in range(0,h.shape[1],BLOCK)],1)
def block_mass(m):
    z=torch.stack([m[s:min(len(m),s+BLOCK)].sum() for s in range(0,len(m),BLOCK)])
    return z/z.sum().clamp_min(1e-12)

@torch.no_grad()
def prefill(model,old):
    o=model(old,use_cache=True,output_hidden_states=True,return_dict=True)
    return [(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)],o.hidden_states[-1].detach().float().clone()

def gradient_teacher(model,items,recent,target):
    leaves=[]; li=[]
    for k,v in items:
        kk=k.detach().clone().requires_grad_(True); vv=v.detach().clone().requires_grad_(True)
        leaves.append((kk,vv)); li.append((kk,vv))
    seq=torch.cat([recent,target[:,:-1]],1)
    pos=torch.arange(OLD,OLD+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(li),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    r=recent.shape[1]-1
    # logits at recent last token predict target[0]; subsequent positions predict later target tokens
    lg=o.logits[:,r:r+target.shape[1],:].float()
    loss=F.cross_entropy(lg.reshape(-1,lg.shape[-1]),target.reshape(-1)); loss.backward()
    ls=[]
    for k,v in leaves:
        ki=(k.grad.float()*k.detach().float()).sum(-1).abs().mean((0,1))
        vi=(v.grad.float()*v.detach().float()).sum(-1).abs().mean((0,1))
        z=ki+vi; ls.append(z/z.sum().clamp_min(1e-12))
    imp=torch.stack(ls).mean(0); imp=(imp+1e-12).sqrt(); imp=imp/imp.sum()
    val=float(loss.detach()); del o,leaves,li; gc.collect()
    return block_mass(imp),val

def select(items,score,budget):
    need=(budget+BLOCK-1)//BLOCK
    bi=torch.topk(score[0],min(need,score.shape[-1])).indices.tolist()
    idx=[]
    for b in bi: idx+=list(range(b*BLOCK,min(OLD,b*BLOCK+BLOCK)))
    idx=torch.tensor(sorted(idx[:budget]),dtype=torch.long)
    return [(k.index_select(-2,idx).contiguous(),v.index_select(-2,idx).contiguous()) for k,v in items],idx

@torch.no_grad()
def eval_cache(model,items,recent,target):
    seq=torch.cat([recent,target[:,:-1]],1)
    pos=torch.arange(OLD,OLD+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(clone(items)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    r=recent.shape[1]-1
    lg=o.logits[:,r:r+target.shape[1],:].float()
    loss=F.cross_entropy(lg.reshape(-1,lg.shape[-1]),target.reshape(-1))
    pred=lg.argmax(-1)
    return {"nll":float(loss),"token_accuracy":float((pred==target).float().mean())}

def recent_only(items):
    # Empty old cache while retaining original absolute positions for future tokens.
    return [(k[...,:0,:].contiguous(),v[...,:0,:].contiguous()) for k,v in items]

def make_windows(tok,texts,n,offset_seed):
    ids=tok("\n\n".join(x for x in texts if x.strip()),add_special_tokens=False,return_tensors="pt").input_ids[0]
    L=OLD+RECENT+TARGET
    rng=random.Random(offset_seed)
    max_start=max(0,len(ids)-L-1)
    starts=sorted(rng.sample(range(max_start+1),min(n,max_start+1))) if max_start>n else list(range(0,max_start+1,max(1,L)))[:n]
    out=[]
    for s in starts[:n]:
        w=ids[s:s+L]
        out.append((w[:OLD].unsqueeze(0),w[OLD:OLD+RECENT].unsqueeze(0),w[OLD+RECENT:].unsqueeze(0),s))
    return out

def main():
    st=time.time(); print("loading dataset",flush=True)
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1")
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    trainw=make_windows(tok,ds["train"]["text"],TRAIN_N,SEED)
    testw=make_windows(tok,ds["test"]["text"],TEST_N,SEED+1000)
    print("loading model",flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    scorer=Scorer(int(model.config.hidden_size))
    data=[]
    for i,(old,recent,target,off) in enumerate(trainw):
        items,h=prefill(model,old); mass,base=gradient_teacher(model,items,recent,target)
        data.append((block_h(h),mass.unsqueeze(0)))
        print("teacher",i,"offset",off,"full_target_nll",round(base,4),"top",torch.topk(mass,min(5,len(mass))).indices.tolist(),flush=True)
        del items,h; gc.collect()
    opt=torch.optim.AdamW(scorer.parameters(),lr=0.003,weight_decay=1e-4)
    curve=[]
    for ep in range(EPOCHS):
        total=0
        for i in random.Random(SEED+ep).sample(range(len(data)),len(data)):
            h,mass=data[i]; lg=scorer(h)
            loss=F.kl_div(F.log_softmax(lg,-1),mass,reduction="batchmean")
            k=min(6,lg.shape[-1]); top=torch.topk(mass[0],k).indices
            y=torch.zeros_like(lg); y[:,top]=1
            loss=loss+0.35*F.binary_cross_entropy_with_logits(lg,y,pos_weight=torch.tensor(max(1.0,(lg.shape[-1]-k)/k)))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=float(loss)
        if ep%20==0 or ep==EPOCHS-1:
            curve.append({"epoch":ep,"loss":total/len(data)}); print("epoch",ep,round(total/len(data),5),flush=True)
    scorer.eval()
    rows=[]
    for i,(old,recent,target,off) in enumerate(testw):
        items,h=prefill(model,old); pred=scorer(block_h(h)).detach()
        full=eval_cache(model,items,recent,target)
        empty=eval_cache(model,recent_only(items),recent,target)
        tmass,_=gradient_teacher(model,items,recent,target); ts=tmass.unsqueeze(0)
        budgets={}
        for b in BUDGETS:
            li,lidx=select(items,pred,b); ti,tidx=select(items,ts,b)
            budgets[str(b)]={"learned":eval_cache(model,li,recent,target),"teacher":eval_cache(model,ti,recent,target),
                             "learned_idx":lidx.tolist(),"teacher_idx":tidx.tolist()}
        rows.append({"offset":off,"full":full,"recent_only":empty,"budgets":budgets})
        print("eval",i,"full",round(full["nll"],4),"recent",round(empty["nll"],4),
              "l32",round(budgets["32"]["learned"]["nll"],4),"l64",round(budgets["64"]["learned"]["nll"],4),flush=True)
    def av(path):
        vals=[]
        for r in rows:
            x=r
            for k in path:x=x[k]
            vals.append(float(x))
        return sum(vals)/len(vals)
    sm={"full":{"nll":av(["full","nll"]),"acc":av(["full","token_accuracy"])},
        "recent_only":{"nll":av(["recent_only","nll"]),"acc":av(["recent_only","token_accuracy"])}}
    for b in BUDGETS:
        sm[str(b)]={}
        for m in ["learned","teacher"]:
            sm[str(b)][m]={"nll":av(["budgets",str(b),m,"nll"]),"acc":av(["budgets",str(b),m,"token_accuracy"])}
        gap=sm["recent_only"]["nll"]-sm["full"]["nll"]
        sm[str(b)]["learned_recovery_fraction"]=(sm["recent_only"]["nll"]-sm[str(b)]["learned"]["nll"])/gap if abs(gap)>1e-9 else None
    result={"model":MODEL_ID,"dataset":"Salesforce/wikitext wikitext-2-raw-v1","old":OLD,"recent":RECENT,"target":TARGET,
            "train_windows":TRAIN_N,"test_windows":TEST_N,"block":BLOCK,"summary":sm,"rows":rows,"curve":curve,
            "elapsed_seconds":time.time()-st,
            "guardrails":["Learned scorer sees only old-prefix causal hidden states.","Gradient future-loss importance is training/oracle supervision only.",
                          "Test windows come from WikiText-2 test split; training windows from train split."]}
    OUT.write_text(json.dumps(result,indent=2)); print("SUMMARY",json.dumps(sm),flush=True)

if __name__=="__main__": main()
