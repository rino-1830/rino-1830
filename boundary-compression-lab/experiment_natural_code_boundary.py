import gc, json, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ.get("MODEL_ID","prism-ml/Bonsai-1.7B-unpacked")
OUT=Path(os.environ.get("RESULT_PATH","boundary-compression-lab/results/natural_code_boundary.json"))
OLD=192; RECENT=48; TARGET=16; BLOCK=8; RANK=64
TRAIN_N=int(os.environ.get("TRAIN_WINDOWS","8")); EVAL_N=int(os.environ.get("EVAL_WINDOWS","8"))
EPOCHS=100; LR=0.003; SEED=73; BUDGETS=[32,64,96]
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

def corpus_files():
    root=Path(transformers.__file__).parent
    train=[
      root/"modeling_utils.py", root/"configuration_utils.py",
      root/"tokenization_utils_base.py", root/"modeling_attn_mask_utils.py",
    ]
    evals=[
      root/"generation"/"utils.py", root/"trainer.py",
      root/"integrations"/"integration_utils.py", root/"cache_utils.py",
    ]
    return [p for p in train if p.exists()],[p for p in evals if p.exists()]

def tokenize_files(tok,files):
    docs=[]
    for p in files:
        text=p.read_text(errors="ignore")
        ids=tok(text,add_special_tokens=False).input_ids
        if len(ids)>OLD+RECENT+TARGET+64: docs.append((str(p.name),ids))
    return docs

def windows(docs,n,seed):
    r=random.Random(seed); out=[]
    for i in range(n):
        name,ids=docs[i%len(docs)]
        need=OLD+RECENT+TARGET
        lo=0; hi=len(ids)-need
        # Spread samples through each file rather than only random neighboring offsets.
        frac=((i//len(docs))+1)/(max(2,(n//len(docs))+2))
        base=int(hi*frac)
        jitter=r.randint(-min(512,base),min(512,max(0,hi-base))) if hi>0 else 0
        s=max(0,min(hi,base+jitter))
        x=torch.tensor([ids[s:s+need]],dtype=torch.long)
        out.append({"name":name,"start":s,"old":x[:,:OLD],"recent":x[:,OLD:OLD+RECENT],"target":x[:,OLD+RECENT:]})
    return out

@torch.no_grad()
def prefill(model,old):
    o=model(old,use_cache=True,output_hidden_states=True,return_dict=True)
    items=[(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]
    h=o.hidden_states[-1].detach().float().clone()
    return items,h

def block_h(h):
    return torch.stack([h[:,s:min(h.shape[1],s+BLOCK),:].mean(1) for s in range(0,h.shape[1],BLOCK)],1)

def block_mass(x):
    z=torch.stack([x[s:min(x.numel(),s+BLOCK)].sum() for s in range(0,x.numel(),BLOCK)])
    return z/z.sum().clamp_min(1e-12)

def gradient_teacher(model,olditems,recent,target):
    leaves=[]; cache=[]
    for k,v in olditems:
        kk=k.detach().clone().requires_grad_(True); vv=v.detach().clone().requires_grad_(True)
        leaves.append((kk,vv)); cache.append((kk,vv))
    seq=torch.cat([recent,target[:,:-1]],1)
    pos=torch.arange(OLD,OLD+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(cache),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    r=recent.shape[1]
    logits=o.logits[:,r-1:r-1+target.shape[1],:].float()
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))
    loss.backward()
    ls=[]
    for k,v in leaves:
        ks=(k.grad.float()*k.detach().float()).sum(-1).abs().mean((0,1))
        vs=(v.grad.float()*v.detach().float()).sum(-1).abs().mean((0,1))
        z=ks+vs; ls.append(z/z.sum().clamp_min(1e-12))
    imp=torch.stack(ls).mean(0); imp=(imp+1e-12).sqrt(); imp=imp/imp.sum()
    val=float(loss.detach()); del o,cache,leaves; gc.collect()
    return block_mass(imp),val

def select(items,score,budget):
    k=max(1,(budget+BLOCK-1)//BLOCK)
    bi=torch.topk(score[0],min(k,score.shape[-1])).indices
    pos=[]
    for j in bi.tolist(): pos+=list(range(j*BLOCK,min(OLD,j*BLOCK+BLOCK)))
    pos=torch.tensor(sorted(set(pos))[:budget],dtype=torch.long)
    return [(a.index_select(-2,pos).contiguous(),b.index_select(-2,pos).contiguous()) for a,b in items],pos

@torch.no_grad()
def score_target(model,olditems,recent,target):
    seq=torch.cat([recent,target[:,:-1]],1)
    pos=torch.arange(OLD,OLD+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(clone(olditems)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    r=recent.shape[1]
    logits=o.logits[:,r-1:r-1+target.shape[1],:].float()
    return float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1)))

def recent_only(items):
    # Preserve one anchor at OLD-1 so absolute-position continuation stays well-defined.
    return [(k[...,-1:,:].clone(),v[...,-1:,:].clone()) for k,v in items]

def train_scorer(model,train,scorer):
    data=[]; teacher_nll=[]
    for i,w in enumerate(train):
        items,h=prefill(model,w["old"]); mass,nll=gradient_teacher(model,items,w["recent"],w["target"])
        data.append((block_h(h),mass.unsqueeze(0))); teacher_nll.append(nll)
        print("teacher",i,w["name"],w["start"],round(nll,4),torch.topk(mass,min(5,mass.numel())).indices.tolist(),flush=True)
        del items,h; gc.collect()
    opt=torch.optim.AdamW(scorer.parameters(),lr=LR,weight_decay=1e-4); curve=[]
    for ep in range(EPOCHS):
        total=0.0
        for h,mass in data:
            lg=scorer(h); loss=F.kl_div(F.log_softmax(lg,-1),mass,reduction="batchmean")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=float(loss)
        if ep%20==0 or ep==EPOCHS-1:
            curve.append({"epoch":ep,"loss":total/len(data)}); print("epoch",ep,total/len(data),flush=True)
    return curve,teacher_nll

@torch.no_grad()
def evaluate(model,evals,scorer):
    rows=[]
    for i,w in enumerate(evals):
        items,h=prefill(model,w["old"]); pred=scorer(block_h(h)).detach()
        full=score_target(model,items,w["recent"],w["target"])
        recent=score_target(model,recent_only(items),w["recent"],w["target"])
        budgets={}
        for b in BUDGETS:
            sel,pos=select(items,pred,b)
            budgets[str(b)]={"nll":score_target(model,sel,w["recent"],w["target"]),"selected":pos.tolist()}
        rows.append({"file":w["name"],"start":w["start"],"full_nll":full,"recent_nll":recent,"budgets":budgets})
        print("eval",i,w["name"],round(full,4),round(recent,4),{b:round(budgets[str(b)]["nll"],4) for b in BUDGETS},flush=True)
    return rows

def summarize(rows):
    def mean(path):
        vals=[]
        for r in rows:
            x=r
            for k in path:x=x[k]
            vals.append(float(x))
        return sum(vals)/len(vals)
    out={"full_nll":mean(["full_nll"]),"recent_nll":mean(["recent_nll"])}
    for b in BUDGETS:
        out[str(b)]={"nll":mean(["budgets",str(b),"nll"])}
        out[str(b)]["delta_vs_full"]=out[str(b)]["nll"]-out["full_nll"]
        out[str(b)]["improvement_vs_recent"]=out["recent_nll"]-out[str(b)]["nll"]
    return out

def main():
    st=time.time(); print("loading",MODEL_ID,flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    trf,evf=corpus_files(); train=windows(tokenize_files(tok,trf),TRAIN_N,SEED); ev=windows(tokenize_files(tok,evf),EVAL_N,SEED+1)
    scorer=Scorer(int(model.config.hidden_size))
    curve,tn=train_scorer(model,train,scorer); scorer.eval(); rows=evaluate(model,ev,scorer); sm=summarize(rows)
    result={"model":MODEL_ID,"corpus":"installed Hugging Face Transformers source files","old":OLD,"recent":RECENT,"target":TARGET,"block":BLOCK,
      "train_files":[p.name for p in trf],"eval_files":[p.name for p in evf],"training_curve":curve,"teacher_train_nll":tn,
      "heldout":rows,"summary":sm,"elapsed_seconds":time.time()-st,
      "guardrails":["Train and evaluation use disjoint source files.","Selector sees only the old causal prefix.","Future target is used only for gradient teacher labels during scorer training.","No synthetic fact-span labels are used."]}
    OUT.write_text(json.dumps(result,indent=2)); print("summary",json.dumps(sm),flush=True)

if __name__=="__main__":main()
