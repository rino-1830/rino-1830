import gc, json, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ.get("MODEL_ID","prism-ml/Bonsai-1.7B-unpacked")
OUT=Path(os.environ.get("RESULT_PATH","boundary-compression-lab/results/block_extrapolate_512.json"))
RECENT=48; TRAIN_OLD=143; TEST_OLD=512; BLOCK=8
RANK=64; EPOCHS=100; LR=0.003; SEED=int(os.environ.get("SEED","61"))
BUDGETS=[32,48,64,96,128]
OUT.parent.mkdir(parents=True,exist_ok=True)
torch.manual_seed(SEED); random.seed(SEED); torch.set_num_threads(min(4,os.cpu_count() or 1))
LABELS=["ALPHA","BETA","GAMMA","DELTA"]
WORDS=["LANTERN","ORBIT","CEDAR","MICA","NOVA","RAVEN","EMBER","LOTUS","QUARTZ","MAPLE","SOLAR","VIOLET","COMET","FROST","IVORY","SABLE","CORAL","ONYX","AURORA","PINE","AMBER","FLINT","IRIS","LUNAR","BIRCH","OPAL","TERRA","ZEPHYR","CINDER","JADE","KITE"]

def code(i): return f"{WORDS[i%len(WORDS)]}-{4100+((i*379)%5800):04d}"
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

def filler_ids(tok,n,seed):
    texts=[
      "Routine maintenance notes mention weather, storage rooms, cables, roads, schedules, tools, and ordinary measurements. ",
      "The notebook contains unrelated inspections, replacement parts, operating procedures, materials, and daily checks. ",
      "Background observations describe benches, labels, transport, temperatures, doors, lighting, and equipment status. "]
    r=random.Random(seed); ids=[]
    while len(ids)<n+64: ids+=tok(r.choice(texts),add_special_tokens=False).input_ids
    return ids[:n]

def build_table(tok,cid,insert_pos,old_len):
    cs=[code(cid*4+j) for j in range(4)]
    facts=tok("".join(f"{LABELS[j]} = {cs[j]}\n" for j in range(4)),add_special_tokens=False).input_ids
    base=filler_ids(tok,max(0,old_len-len(facts)),cid)
    pos=max(0,min(insert_pos,len(base)))
    old=(base[:pos]+facts+base[pos:])[:old_len]
    if len(old)<old_len: old+=filler_ids(tok,old_len-len(old),999+cid)
    mask=torch.zeros(old_len); mask[pos:min(old_len,pos+len(facts))]=1
    packs=[]
    for q in range(4):
        qids=tok(f" Question: What is the exact identifier for {LABELS[q]}? Answer:",add_special_tokens=False).input_ids
        pad=filler_ids(tok,max(0,RECENT+1-len(qids)),1000+cid*10+q)
        tail=(pad+qids)[-(RECENT+1):]
        tgt=tok(" "+cs[q],add_special_tokens=False).input_ids[:12]
        packs.append({"recent":torch.tensor([tail[:-1]]),"last":torch.tensor([[tail[-1]]]),"t":torch.tensor([tgt]),
                      "meta":{"case_id":cid,"query":LABELS[q],"insert_pos":pos}})
    return torch.tensor([old]),mask.unsqueeze(0),packs

@torch.no_grad()
def prefill(model,ids):
    o=model(ids,use_cache=True,output_hidden_states=True,return_dict=True)
    return [(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)],o.hidden_states[-1].detach().float().clone()

def blockify_h(h):
    return torch.stack([h[:,s:min(h.shape[1],s+BLOCK),:].mean(1) for s in range(0,h.shape[1],BLOCK)],1)
def blockify_mass(m):
    z=torch.stack([m[s:min(m.numel(),s+BLOCK)].sum() for s in range(0,m.numel(),BLOCK)])
    return z/z.sum().clamp_min(1e-12)

def grad_one(model,olditems,pack,old_len):
    leaves=[]; items=[]
    for k,v in olditems:
        kk=k.detach().clone().requires_grad_(True); vv=v.detach().clone().requires_grad_(True)
        leaves.append((kk,vv)); items.append((kk,vv))
    seq=torch.cat([pack["recent"],pack["last"],pack["t"][:,:-1]],1)
    pos=torch.arange(old_len,old_len+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(items),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    r=pack["recent"].shape[1]; lg=o.logits[:,r:r+pack["t"].shape[1],:].float()
    loss=F.cross_entropy(lg.reshape(-1,lg.shape[-1]),pack["t"].reshape(-1)); loss.backward()
    ls=[]
    for k,v in leaves:
        ks=(k.grad.float()*k.detach().float()).sum(-1).abs().mean((0,1))
        vs=(v.grad.float()*v.detach().float()).sum(-1).abs().mean((0,1))
        z=ks+vs; ls.append(z/z.sum().clamp_min(1e-12))
    imp=torch.stack(ls).mean(0); imp=imp/imp.sum().clamp_min(1e-12)
    val=float(loss.detach()); del o,items,leaves; gc.collect()
    return imp.detach(),val

def teacher(model,olditems,packs,old_len):
    agg=None
    for p in packs:
        x,_=grad_one(model,olditems,p,old_len); agg=x if agg is None else agg+x
    agg=agg/len(packs); agg=(agg+1e-12).sqrt(); agg=agg/agg.sum()
    return blockify_mass(agg)

def train(model,tok,scorer):
    data=[]; posgrid=[0,16,32,48,64,80,96]
    for cid in range(10):
        old,_,packs=build_table(tok,cid,posgrid[cid%len(posgrid)],TRAIN_OLD)
        items,h=prefill(model,old); mass=teacher(model,items,packs,TRAIN_OLD)
        data.append((blockify_h(h),mass.unsqueeze(0))); print("teacher",cid,flush=True)
        del items,h; gc.collect()
    opt=torch.optim.AdamW(scorer.parameters(),lr=LR,weight_decay=1e-4)
    for ep in range(EPOCHS):
        order=list(range(len(data))); random.Random(SEED+ep).shuffle(order)
        for i in order:
            h,mass=data[i]; lg=scorer(h); logp=F.log_softmax(lg,-1)
            kl=F.kl_div(logp,mass,reduction="batchmean")
            k=min(6,lg.shape[-1]); idx=torch.topk(mass[0],k).indices
            binary=torch.zeros_like(lg); binary[:,idx]=1
            pw=torch.tensor(max(1.0,(lg.shape[-1]-k)/max(1,k)))
            loss=kl+0.35*F.binary_cross_entropy_with_logits(lg,binary,pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep%20==0: print("epoch",ep,flush=True)

def positions_for_blocks(score,budget):
    need=max(1,(budget+BLOCK-1)//BLOCK)
    bi=torch.topk(score[0],min(need,score.shape[-1])).indices
    keep=[]
    for j in bi.tolist():
        keep+=list(range(j*BLOCK,min(TEST_OLD,j*BLOCK+BLOCK)))
        if len(keep)>=budget: break
    return torch.tensor(sorted(keep[:budget]),dtype=torch.long)

def span_oracle(mask,budget):
    fact=torch.where(mask[0]>0.5)[0].tolist()[:budget]
    if len(fact)<budget:
        used=set(fact); fact += [i for i in range(TEST_OLD) if i not in used][:budget-len(fact)]
    return torch.tensor(sorted(fact),dtype=torch.long)
def uniform(budget):
    return torch.linspace(0,TEST_OLD-1,budget).round().long().unique()
def select(items,pos): return [(k.index_select(-2,pos).contiguous(),v.index_select(-2,pos).contiguous()) for k,v in items]

@torch.no_grad()
def extend(model,items,recent):
    pos=torch.arange(TEST_OLD,TEST_OLD+recent.shape[1],dtype=torch.long)
    o=model(recent,past_key_values=dyn(clone(items)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    return [(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]
@torch.no_grad()
def score(model,p,items):
    seq=torch.cat([p["last"],p["t"][:,:-1]],1); st=TEST_OLD+RECENT
    pos=torch.arange(st,st+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(clone(items)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    lg=o.logits.float(); loss=F.cross_entropy(lg.reshape(-1,lg.shape[-1]),p["t"].reshape(-1)); pr=lg.argmax(-1)
    return {"nll":float(loss),"exact":bool((pr==p["t"]).all()),"acc":float((pr==p["t"]).float().mean())}

def main():
    st=time.time(); print("loading",MODEL_ID,flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(MODEL_ID); scorer=BlockScorer(int(model.config.hidden_size))
    train(model,tok,scorer); scorer.eval()
    rows=[]; positions=[16,112,240,368,448]
    for z,cid in enumerate(range(60,65)):
        old,mask,packs=build_table(tok,cid,positions[z],TEST_OLD); items,h=prefill(model,old); pred=scorer(blockify_h(h)).detach()
        for p in packs:
            fullitems=extend(model,items,p["recent"]); fm=score(model,p,fullitems)
            recent=[(k[...,TEST_OLD:,:],v[...,TEST_OLD:,:]) for k,v in fullitems]; rm=score(model,p,recent)
            ms={}
            for b in BUDGETS:
                lp=positions_for_blocks(pred,b); op=span_oracle(mask,b); up=uniform(b)
                ms[str(b)]={
                    "learned":{"metrics":score(model,p,extend(model,select(items,lp),p["recent"])),"selected":lp.tolist()},
                    "oracle":{"metrics":score(model,p,extend(model,select(items,op),p["recent"]))},
                    "uniform":{"metrics":score(model,p,extend(model,select(items,up),p["recent"]))}}
            rows.append({"meta":p["meta"],"full":fm,"recent":rm,"budgets":ms})
        print("eval",cid,positions[z],flush=True)
    def avg(path):
        vals=[]
        for r in rows:
            x=r
            for k in path:x=x[k]
            vals.append(float(x))
        return sum(vals)/len(vals)
    sm={"full":{"nll":avg(["full","nll"]),"exact":avg(["full","exact"])},
        "recent":{"nll":avg(["recent","nll"]),"exact":avg(["recent","exact"])}}
    for b in BUDGETS:
        sm[str(b]]={}
    for b in BUDGETS:
        sm[str(b)]={}
        for m in ["learned","oracle","uniform"]:
            sm[str(b)][m]={"nll":avg(["budgets",str(b),m,"metrics","nll"]),"exact":avg(["budgets",str(b),m,"metrics","exact"])}
    result={"model":MODEL_ID,"train_old":TRAIN_OLD,"test_old":TEST_OLD,"recent":RECENT,"block":BLOCK,
            "summary":sm,"rows":rows,"elapsed":time.time()-st,
            "guardrails":["Scorer training sees only 143-token old prefixes.","512-token evaluation uses no gradient teacher or future query for learned selection.","Held-out facts use disjoint identifiers and distant offsets."]}
    OUT.write_text(json.dumps(result,indent=2)); print(json.dumps(sm,indent=2),flush=True); print("wrote",OUT,flush=True)
if __name__=="__main__": main()
