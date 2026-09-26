import gc, json, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID=os.environ.get("MODEL_ID","prism-ml/Bonsai-1.7B-unpacked")
OUT=Path(os.environ.get("RESULT_PATH","boundary-compression-lab/results/gradient_block_boundary.json"))
CKPT=Path(os.environ.get("CKPT_PATH","boundary-compression-lab/results/gradient_block_boundary.pt"))
RECENT=int(os.environ.get("RECENT_TOKENS","48"))
OLD_LEN=int(os.environ.get("OLD_TOKENS","143"))
BLOCK=int(os.environ.get("BLOCK_SIZE","8"))
EPOCHS=int(os.environ.get("SCORER_EPOCHS","100"))
RANK=int(os.environ.get("SCORER_RANK","64"))
LR=float(os.environ.get("LR","0.003"))
TRAIN_TABLES=int(os.environ.get("TRAIN_TABLES","10"))
BUDGETS=[16,24,32,48,64]
SEED=37
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
    def forward(self,block_h): return self.net(block_h.float()).squeeze(-1)

def filler_ids(tok,n,seed):
    texts=[
      "Routine maintenance notes mention weather, storage rooms, cables, roads, schedules, tools, and ordinary measurements. ",
      "The notebook contains unrelated inspections, replacement parts, operating procedures, materials, and daily checks. ",
      "Background observations describe benches, labels, transport, temperatures, doors, lighting, and equipment status. "]
    r=random.Random(seed); ids=[]
    while len(ids)<n+64: ids+=tok(r.choice(texts),add_special_tokens=False).input_ids
    return ids[:n]

def build_table(tok,cid,insert_pos):
    cs=[code(cid*4+j) for j in range(4)]
    fact_text="".join(f"{LABELS[j]} = {cs[j]}\n" for j in range(4))
    facts=tok(fact_text,add_special_tokens=False).input_ids
    base=filler_ids(tok,max(0,OLD_LEN-len(facts)),cid)
    pos=max(0,min(insert_pos,len(base)))
    old=(base[:pos]+facts+base[pos:])[:OLD_LEN]
    if len(old)<OLD_LEN: old+=filler_ids(tok,OLD_LEN-len(old),999+cid)
    mask=torch.zeros(OLD_LEN); mask[pos:min(OLD_LEN,pos+len(facts))]=1
    packs=[]
    for q in range(4):
        qids=tok(f" Question: What is the exact identifier for {LABELS[q]}? Answer:",add_special_tokens=False).input_ids
        pad=filler_ids(tok,max(0,RECENT+1-len(qids)),1000+cid*10+q)
        tail=(pad+qids)[-(RECENT+1):]
        tgt=tok(" "+cs[q],add_special_tokens=False).input_ids[:12]
        packs.append({"recent":torch.tensor([tail[:-1]]),"last":torch.tensor([[tail[-1]]]),"t":torch.tensor([tgt]),
                      "meta":{"case_id":cid,"query":LABELS[q],"answer":cs[q],"insert_pos":pos,"fact_tokens":len(facts)}})
    return torch.tensor([old]),mask.unsqueeze(0),packs

@torch.no_grad()
def prefill(model,ids):
    o=model(ids,use_cache=True,output_hidden_states=True,return_dict=True)
    return [(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)],o.hidden_states[-1].detach().float().clone()

def blockify_h(h):
    xs=[]
    for s in range(0,h.shape[1],BLOCK): xs.append(h[:,s:min(h.shape[1],s+BLOCK),:].mean(dim=1))
    return torch.stack(xs,dim=1)

def blockify_mass(m):
    xs=[]
    for s in range(0,m.numel(),BLOCK): xs.append(m[s:min(m.numel(),s+BLOCK)].sum())
    z=torch.stack(xs); return z/z.sum().clamp_min(1e-12)

def grad_one(model,olditems,pack):
    leaves=[]; items=[]
    for k,v in olditems:
        kk=k.detach().clone().requires_grad_(True); vv=v.detach().clone().requires_grad_(True)
        leaves.append((kk,vv)); items.append((kk,vv))
    seq=torch.cat([pack["recent"],pack["last"],pack["t"][:,:-1]],1)
    pos=torch.arange(OLD_LEN,OLD_LEN+seq.shape[1],dtype=torch.long)
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

def teacher(model,olditems,packs):
    agg=None; losses=[]
    for p in packs:
        x,l=grad_one(model,olditems,p); agg=x if agg is None else agg+x; losses.append(l)
    agg=agg/len(packs); agg=(agg+1e-12).sqrt(); agg=agg/agg.sum()
    return blockify_mass(agg),losses

def positions_for_blocks(block_idx,budget):
    keep=[]
    for j in block_idx.tolist():
        s=j*BLOCK; keep+=list(range(s,min(OLD_LEN,s+BLOCK)))
        if len(keep)>=budget: break
    return torch.tensor(sorted(keep[:budget]),dtype=torch.long)

def select_by_score(items,score,budget):
    need=max(1,(budget+BLOCK-1)//BLOCK)
    idx=torch.topk(score[0],min(need,score.shape[-1])).indices
    pos=positions_for_blocks(idx,budget)
    return [(k.index_select(-2,pos).contiguous(),v.index_select(-2,pos).contiguous()) for k,v in items],pos

def span_oracle(mask,budget):
    fact=torch.where(mask[0]>0.5)[0].tolist()
    fact=fact[:budget]
    if len(fact)<budget:
        used=set(fact); fact += [i for i in range(OLD_LEN) if i not in used][:budget-len(fact)]
    return torch.tensor(sorted(fact),dtype=torch.long)

def select_pos(items,pos): return [(k.index_select(-2,pos).contiguous(),v.index_select(-2,pos).contiguous()) for k,v in items]

@torch.no_grad()
def extend(model,olditems,recent):
    pos=torch.arange(OLD_LEN,OLD_LEN+recent.shape[1],dtype=torch.long)
    o=model(recent,past_key_values=dyn(clone(olditems)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    return [(k.detach().clone(),v.detach().clone()) for k,v in legacy(o.past_key_values)]

@torch.no_grad()
def score(model,pack,items):
    seq=torch.cat([pack["last"],pack["t"][:,:-1]],1); st=OLD_LEN+RECENT
    pos=torch.arange(st,st+seq.shape[1],dtype=torch.long)
    o=model(seq,past_key_values=dyn(clone(items)),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    lg=o.logits.float(); loss=F.cross_entropy(lg.reshape(-1,lg.shape[-1]),pack["t"].reshape(-1)); pr=lg.argmax(-1)
    return {"nll":float(loss),"token_accuracy":float((pr==pack["t"]).float().mean()),"exact":bool((pr==pack["t"]).all())}

def build_dataset(model,tok):
    data=[]; positions=[0,16,32,48,64,80,96]
    for cid in range(TRAIN_TABLES):
        old,mask,packs=build_table(tok,cid,positions[cid%len(positions)])
        items,h=prefill(model,old); mass,losses=teacher(model,items,packs)
        data.append((blockify_h(h),mass.unsqueeze(0)))
        print("teacher",cid,"loss",[round(x,3) for x in losses],"top",torch.topk(mass,min(6,len(mass))).indices.tolist(),flush=True)
        del items,h; gc.collect()
    return data

def train(scorer,data):
    opt=torch.optim.AdamW(scorer.parameters(),lr=LR,weight_decay=1e-4); curve=[]
    for ep in range(EPOCHS):
        order=list(range(len(data))); random.Random(SEED+ep).shuffle(order); total=0
        for i in order:
            h,mass=data[i]; lg=scorer(h); logp=F.log_softmax(lg,dim=-1)
            kl=F.kl_div(logp,mass,reduction="batchmean")
            k=min(6,lg.shape[-1]); idx=torch.topk(mass[0],k).indices
            binary=torch.zeros_like(lg); binary[:,idx]=1
            pw=torch.tensor(max(1.0,(lg.shape[-1]-k)/max(1,k)))
            bce=F.binary_cross_entropy_with_logits(lg,binary,pos_weight=pw)
            loss=kl+0.35*bce; opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=float(loss)
        if ep%20==0 or ep==EPOCHS-1:
            rec={"epoch":ep,"loss":total/len(data)}; curve.append(rec); print("train",rec,flush=True)
    return curve

def evaluate(model,tok,scorer):
    rows=[]; testpos=[8,24,40,56,72,88]
    for z,cid in enumerate(range(30,36)):
        old,mask,packs=build_table(tok,cid,testpos[z]); items,h=prefill(model,old)
        pred=scorer(blockify_h(h)).detach(); tmass,_=teacher(model,items,packs); tscore=tmass.unsqueeze(0)
        for p in packs:
            fullitems=extend(model,items,p["recent"]); fm=score(model,p,fullitems)
            recent=[(k[...,OLD_LEN:,:],v[...,OLD_LEN:,:]) for k,v in fullitems]; rm=score(model,p,recent)
            bs={}
            for b in BUDGETS:
                lp,lpos=select_by_score(items,pred,b); lm=score(model,p,extend(model,lp,p["recent"]))
                tp,tpos=select_by_score(items,tscore,b); tm=score(model,p,extend(model,tp,p["recent"]))
                op=span_oracle(mask,b); om=score(model,p,extend(model,select_pos(items,op),p["recent"]))
                bs[str(b)]={"learned_block":{"metrics":lm,"selected":lpos.tolist()},
                            "teacher_block":{"metrics":tm,"selected":tpos.tolist()},
                            "span_oracle":{"metrics":om,"selected":op.tolist()}}
            rows.append({"meta":p["meta"],"full":fm,"recent_only":rm,"budgets":bs})
        print("eval",cid,"pos",testpos[z],flush=True)
    return rows

def summarize(rows):
    out={"full":{"avg_nll":sum(r["full"]["nll"] for r in rows)/len(rows),"exact_rate":sum(r["full"]["exact"] for r in rows)/len(rows)},
         "recent_only":{"avg_nll":sum(r["recent_only"]["nll"] for r in rows)/len(rows),"exact_rate":sum(r["recent_only"]["exact"] for r in rows)/len(rows)}}
    for b in BUDGETS:
        out[str(b)]={}
        for m in ["learned_block","teacher_block","span_oracle"]:
            xs=[r["budgets"][str(b)][m]["metrics"] for r in rows]
            out[str(b)][m]={"avg_nll":sum(x["nll"] for x in xs)/len(xs),"exact_rate":sum(x["exact"] for x in xs)/len(xs)}
    return out

def main():
    st=time.time(); print("loading",MODEL_ID,flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(MODEL_ID); scorer=BlockScorer(int(model.config.hidden_size))
    data=build_dataset(model,tok); curve=train(scorer,data); scorer.eval(); rows=evaluate(model,tok,scorer); sm=summarize(rows)
    result={"model":MODEL_ID,"method":"random-position self-supervised block scorer distilled from future-loss KV gradients",
            "old_tokens":OLD_LEN,"recent_tokens":RECENT,"block_size":BLOCK,"budgets":BUDGETS,
            "training_curve":curve,"heldout":rows,"summary":sm,"elapsed_seconds":time.time()-st,
            "guardrails":["Future targets are used only to make gradient teacher labels during training/evaluation oracle.",
                          "Learned block scorer sees only causal old-prefix hidden states at inference.",
                          "Fact spans use unseen offsets at evaluation.","Streaming compression occurs before recent/query tokens are processed."]}
    OUT.write_text(json.dumps(result,indent=2)); torch.save({"state_dict":scorer.state_dict()},CKPT)
    print("summary",json.dumps(sm),flush=True); print("wrote",OUT,flush=True)

if __name__=="__main__": main()
