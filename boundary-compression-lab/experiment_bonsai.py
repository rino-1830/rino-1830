import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = os.environ.get('MODEL_ID', 'prism-ml/Bonsai-1.7B-unpacked')
DTYPE_NAME = os.environ.get('DTYPE', 'float32')
DTYPE = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DTYPE_NAME]
OUT = Path(os.environ.get('RESULT_PATH', 'boundary-compression-lab/results/bonsai_boundary_results.json'))
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(0)

def legacy(cache):
    if hasattr(cache, 'to_legacy_cache'):
        return cache.to_legacy_cache()
    return tuple(cache)

def dynamic_from_legacy(items):
    if hasattr(DynamicCache, 'from_legacy_cache'):
        return DynamicCache.from_legacy_cache(tuple(items))
    c = DynamicCache()
    for i, (k, v) in enumerate(items):
        c.update(k, v, i)
    return c

def clone_cache(cache):
    return dynamic_from_legacy([(k.clone(), v.clone()) for k, v in legacy(cache)])

def recent_cache(cache, budget):
    out = []
    for k, v in legacy(cache):
        b = min(budget, k.shape[-2])
        out.append((k[..., -b:, :].contiguous(), v[..., -b:, :].contiguous()))
    return dynamic_from_legacy(out)

def mean_pool_cache(cache, budget):
    out = []
    for k, v in legacy(cache):
        n = k.shape[-2]
        if budget >= n:
            out.append((k.clone(), v.clone()))
            continue
        shape = k.shape
        kk = k.transpose(-1, -2).reshape(-1, shape[-1], n)
        vv = v.transpose(-1, -2).reshape(-1, shape[-1], n)
        kk = F.adaptive_avg_pool1d(kk.float(), budget).to(k.dtype)
        vv = F.adaptive_avg_pool1d(vv.float(), budget).to(v.dtype)
        kk = kk.reshape(*shape[:-2], shape[-1], budget).transpose(-1, -2).contiguous()
        vv = vv.reshape(*shape[:-2], shape[-1], budget).transpose(-1, -2).contiguous()
        out.append((kk, vv))
    return dynamic_from_legacy(out)

def attention_boundary_cache(cache, attentions, budget, observation=16):
    out = []
    for layer_idx, (k, v) in enumerate(legacy(cache)):
        n = k.shape[-2]
        if budget >= n:
            out.append((k.clone(), v.clone()))
            continue
        obs = min(observation, budget, n)
        old_end = n - obs
        old_budget = budget - obs
        recent_idx = torch.arange(old_end, n, device=k.device)
        if old_budget <= 0 or old_end <= 0:
            idx = recent_idx[-budget:]
        else:
            a = attentions[layer_idx]
            if a is None:
                raise RuntimeError('attention_boundary requires eager attention')
            q0 = max(0, a.shape[-2] - obs)
            score = a[..., q0:, :old_end].float().mean(dim=(0, 1, 2))
            take = min(old_budget, old_end)
            top = torch.topk(score, k=take, largest=True, sorted=False).indices
            idx = torch.cat([torch.sort(top).values, recent_idx])
        out.append((k.index_select(-2, idx).contiguous(), v.index_select(-2, idx).contiguous()))
    return dynamic_from_legacy(out)

def cache_bytes(cache):
    return sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in legacy(cache))

def make_retrieval_case(code, filler_repeats=18):
    filler = (
        'A field notebook records observations about weather, tools, roads, materials, and schedules. '
        'Each sentence is unrelated to the identifier above and should not replace it in memory. '
        'The notes mention ordinary measurements, maintenance work, and routine checks. '
    ) * filler_repeats
    prefix = (
        f'Remember this exact identifier for the question at the end: KEY = {code}.\n\n'
        f'{filler}\nQuestion: What is the exact KEY?\nAnswer:'
    )
    return prefix, f' {code}'

NATURAL_CASES = [(
    'A transformer language model predicts the next token from a sequence of earlier tokens. '
    'During autoregressive inference, attention keys and values from earlier positions are cached '
    'so they do not need to be recomputed. As the context grows, this cache can become a large '
    'fraction of total memory use. One possible strategy is to preserve only the information from '
    'the distant prefix that is useful for predicting the continuation. In that setting, the main '
    'question is whether a small boundary state can replace most of the old cache without changing ',
    'the model prediction very much.'
)]

def encode_pair(tok, prefix, target, max_prefix=256, max_target=16):
    p = tok(prefix, add_special_tokens=False, return_tensors='pt').input_ids
    t = tok(target, add_special_tokens=False, return_tensors='pt').input_ids
    if p.shape[1] > max_prefix:
        left = max_prefix // 2
        p = torch.cat([p[:, :left], p[:, -(max_prefix-left):]], dim=1)
    return p, t[:, :max_target]

@torch.inference_mode()
def score_suffix(model, base_cache, prefix_len, target_ids):
    if target_ids.shape[1] < 2:
        return {'nll': float('nan'), 'tokens': 0}
    cache = clone_cache(base_cache)
    losses = []
    for j in range(target_ids.shape[1]-1):
        pos = prefix_len + j
        out = model(
            target_ids[:, j:j+1],
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([pos], dtype=torch.long),
            position_ids=torch.tensor([[pos]], dtype=torch.long),
            return_dict=True,
        )
        cache = out.past_key_values
        losses.append(F.cross_entropy(out.logits[:, -1, :].float(), target_ids[:, j+1], reduction='none'))
    return {'nll': torch.cat(losses).mean().item(), 'tokens': len(losses)}

@torch.inference_mode()
def compare_case(model, tok, prefix, target, budgets=(16,32,64), observation=16):
    prefix_ids, target_ids = encode_pair(tok, prefix, target)
    t0=time.time()
    out = model(prefix_ids, use_cache=True, output_attentions=True, return_dict=True)
    prefill_s=time.time()-t0
    full_cache=out.past_key_values
    prefix_len=prefix_ids.shape[1]
    first_nll=F.cross_entropy(out.logits[:, -1, :].float(), target_ids[:,0]).item()
    full=score_suffix(model, full_cache, prefix_len, target_ids)
    full_nll=(first_nll+full['nll']*full['tokens'])/max(1,1+full['tokens'])
    rows=[{'method':'full','budget':prefix_len,'prefix_tokens':prefix_len,'target_tokens':int(target_ids.shape[1]),
           'nll':full_nll,'delta_nll':0.0,'cache_bytes_runtime_dtype':cache_bytes(full_cache)}]
    for b in budgets:
        if b>=prefix_len: continue
        variants={
            'recent':recent_cache(full_cache,b),
            'mean_pool':mean_pool_cache(full_cache,b),
            'attention_boundary':attention_boundary_cache(full_cache,out.attentions,b,observation),
        }
        for name,c in variants.items():
            s=score_suffix(model,c,prefix_len,target_ids)
            nll=(first_nll+s['nll']*s['tokens'])/max(1,1+s['tokens'])
            rows.append({'method':name,'budget':b,'prefix_tokens':prefix_len,'target_tokens':int(target_ids.shape[1]),
                         'nll':nll,'delta_nll':nll-full_nll,'cache_bytes_runtime_dtype':cache_bytes(c)})
    return {'target':target,'prefix_tokens':prefix_len,'target_tokens':int(target_ids.shape[1]),
            'prefill_seconds':prefill_s,'rows':rows}

def theoretical_kv_bytes(config,tokens,bytes_per_element=2):
    return config.num_hidden_layers*config.num_key_value_heads*config.head_dim*2*tokens*bytes_per_element

def main():
    started=time.time()
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    print('loading model',MODEL_ID,DTYPE_NAME,flush=True)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,torch_dtype=DTYPE,low_cpu_mem_usage=True,attn_implementation='eager')
    model.eval()
    cases=[
        ('retrieval_A',)+make_retrieval_case('LANTERN-4827'),
        ('retrieval_B',)+make_retrieval_case('ORBIT-7319'),
        ('natural_1',NATURAL_CASES[0][0],NATURAL_CASES[0][1]),
    ]
    results=[]
    for name,prefix,target in cases:
        print('case',name,flush=True)
        r=compare_case(model,tok,prefix,target); r['name']=name; results.append(r)
        for row in r['rows']:
            print(name,row['method'],row['budget'],'delta',round(row['delta_nll'],4),flush=True)
    cfg=model.config
    max_ctx=int(getattr(cfg,'max_position_embeddings',32768))
    full_kv=theoretical_kv_bytes(cfg,max_ctx,2)
    q1_weight_bytes=248*1024*1024
    scenarios=[]
    for budget in [512,1024,2048,4096,max_ctx]:
        if budget>max_ctx: continue
        kv=theoretical_kv_bytes(cfg,budget,2)
        scenarios.append({'retained_kv_tokens':budget,'fp16_kv_bytes':kv,
                          'q1_weight_plus_fp16_kv_bytes':q1_weight_bytes+kv,
                          'ratio_vs_q1_weight_plus_full_fp16_kv':(q1_weight_bytes+full_kv)/(q1_weight_bytes+kv)})
    doc={'model':MODEL_ID,'runtime_dtype':DTYPE_NAME,
         'config':{'layers':cfg.num_hidden_layers,'kv_heads':cfg.num_key_value_heads,'head_dim':cfg.head_dim,
                   'max_position_embeddings':max_ctx},
         'published_q1_gguf_approx_bytes':q1_weight_bytes,
         'fp16_kv_bytes_per_token':theoretical_kv_bytes(cfg,1,2),
         'memory_scenarios':scenarios,'cases':results,'elapsed_seconds':time.time()-started,
         'notes':['HF checkpoint is the unpacked FP16 representation of the same 1-bit Bonsai-1.7B weights.',
                  'attention_boundary selects old KV positions using attention from the final observation window; it is not yet a learned latent encoder.',
                  'mean_pool is a naive control because RoPE-rotated keys are not generally safe to average.']}
    OUT.write_text(json.dumps(doc,indent=2))
    print('wrote',OUT,flush=True)

if __name__=='__main__':
    main()
