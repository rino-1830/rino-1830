import gc
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = os.environ.get('MODEL_ID', 'prism-ml/Bonsai-1.7B-unpacked')
OUT = Path(os.environ.get('RESULT_PATH', 'boundary-compression-lab/results/gradient_teacher.json'))
CKPT = Path(os.environ.get('CKPT_PATH', 'boundary-compression-lab/results/gradient_teacher.pt'))
RECENT = int(os.environ.get('RECENT_TOKENS', '48'))
MAX_PREFIX = int(os.environ.get('MAX_PREFIX', '192'))
RANK = int(os.environ.get('SCORER_RANK', '64'))
EPOCHS = int(os.environ.get('SCORER_EPOCHS', '120'))
LR = float(os.environ.get('LR', '0.003'))
THREADS = int(os.environ.get('TORCH_THREADS', '4'))
TRAIN_TABLES = int(os.environ.get('TRAIN_TABLES', '10'))
BUDGETS = [int(x) for x in os.environ.get('BUDGETS', '8,16,24,32,48,64').split(',')]
SEED = 23

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_num_threads(min(THREADS, os.cpu_count() or 1))

LABELS = ['ALPHA', 'BETA', 'GAMMA', 'DELTA']
WORDS = ['LANTERN','ORBIT','CEDAR','MICA','NOVA','RAVEN','EMBER','LOTUS',
         'QUARTZ','MAPLE','SOLAR','VIOLET','COMET','FROST','IVORY','DELTA',
         'SABLE','CORAL','ONYX','AURORA','PINE','AMBER','FLINT','IRIS',
         'LUNAR','BIRCH','OPAL','TERRA','ZEPHYR','CINDER','JADE','KITE']

def code(i):
    return f'{WORDS[i % len(WORDS)]}-{4100 + ((i * 379) % 5800):04d}'

def filler(seed, repeats=10):
    variants = [
        'A field notebook records routine observations about tools, roads, materials, weather, and schedules. ',
        'The maintenance log lists ordinary checks, measurements, replacement parts, and inspection notes. ',
        'Unrelated notes describe storage rooms, work benches, cables, labels, and daily operating procedures. ',
        'The report contains mundane details that should not replace the identifiers stored in the table. ',
    ]
    r = random.Random(seed)
    return ''.join(r.choice(variants) for _ in range(repeats))

def case(case_id, q):
    cs = [code(case_id * 4 + j) for j in range(4)]
    table = '\n'.join(f'{LABELS[j]} = {cs[j]}' for j in range(4))
    prefix = ('Memorize the following identifier table exactly.\n' + table + '\n\n' + filler(case_id)
              + f'\nQuestion: What is the exact identifier for {LABELS[q]}?\nAnswer:')
    return prefix, ' ' + cs[q], {'case_id': case_id, 'query': LABELS[q], 'answer': cs[q]}

def encode(tok, prefix, target):
    p = tok(prefix, add_special_tokens=False, return_tensors='pt').input_ids
    t = tok(target, add_special_tokens=False, return_tensors='pt').input_ids[:, :12]
    if p.shape[1] > MAX_PREFIX:
        p = torch.cat([p[:, :MAX_PREFIX-RECENT], p[:, -RECENT:]], dim=1)
    return p, t

def legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache, 'to_legacy_cache') else tuple(cache)

def dyn(items):
    c = DynamicCache()
    for i, (k, v) in enumerate(items):
        c.update(k, v, i)
    return c

def clone_items(items):
    return [(k.clone(), v.clone()) for k, v in items]

class Scorer(nn.Module):
    def __init__(self, hidden, rank=RANK):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, rank), nn.Tanh(), nn.Linear(rank, 1))
    def forward(self, x):
        return self.net(x.float()).squeeze(-1)

def prepare_table(tok, case_id):
    packs = []
    old_ids_common = None
    for q in range(4):
        prefix, target, meta = case(case_id, q)
        p, t = encode(tok, prefix, target)
        cached = p[:, :-1]
        old_len = max(1, cached.shape[1] - RECENT)
        old_ids = cached[:, :old_len]
        recent = cached[:, old_len:]
        if old_ids_common is None:
            old_ids_common = old_ids
        elif not torch.equal(old_ids_common, old_ids):
            raise RuntimeError('old prefix differs across future queries')
        packs.append({'p': p, 't': t, 'recent': recent, 'last': p[:, -1:],
                      'old_len': int(old_len), 'cached_len': int(cached.shape[1]), 'meta': meta})
    return old_ids_common, packs

@torch.no_grad()
def prefill_old(model, old_ids):
    o = model(old_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    items = [(k.detach().clone(), v.detach().clone()) for k, v in legacy(o.past_key_values)]
    h = o.hidden_states[-1].detach().float().clone()
    return items, h

def grad_importance_one_query(model, old_items, pack):
    leaves = []
    cache_items = []
    for k, v in old_items:
        kk = k.detach().clone().requires_grad_(True)
        vv = v.detach().clone().requires_grad_(True)
        leaves.append((kk, vv))
        cache_items.append((kk, vv))
    cache = dyn(cache_items)
    t = pack['t']
    seq = torch.cat([pack['recent'], pack['last'], t[:, :-1]], dim=1)
    start = pack['old_len']
    pos = torch.arange(start, start + seq.shape[1], dtype=torch.long)
    out = model(seq, past_key_values=cache, position_ids=pos.unsqueeze(0), cache_position=pos,
                use_cache=False, return_dict=True)
    r = pack['recent'].shape[1]
    logits = out.logits[:, r:r+t.shape[1], :].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1))
    loss.backward()
    layer_scores = []
    for kk, vv in leaves:
        if kk.grad is None or vv.grad is None:
            raise RuntimeError('cache gradient missing')
        ks = (kk.grad.float() * kk.detach().float()).sum(dim=-1).abs().mean(dim=(0, 1))
        vs = (vv.grad.float() * vv.detach().float()).sum(dim=-1).abs().mean(dim=(0, 1))
        s = ks + vs
        s = s / s.sum().clamp_min(1e-12)
        layer_scores.append(s)
    imp = torch.stack(layer_scores).mean(dim=0)
    imp = imp / imp.sum().clamp_min(1e-12)
    del out, cache, leaves, cache_items
    gc.collect()
    return imp.detach(), float(loss.detach().item())

def gradient_teacher(model, old_items, packs):
    agg = None
    losses = []
    for pack in packs:
        imp, loss = grad_importance_one_query(model, old_items, pack)
        agg = imp if agg is None else agg + imp
        losses.append(loss)
    agg = agg / len(packs)
    mass = (agg + 1e-12).sqrt()
    mass = mass / mass.sum()
    return mass, losses

def build_dataset(model, tok):
    data = []
    print('building gradient-deletion teacher', flush=True)
    for cid in range(TRAIN_TABLES):
        old, packs = prepare_table(tok, cid)
        old_items, h = prefill_old(model, old)
        mass, losses = gradient_teacher(model, old_items, packs)
        top = torch.topk(mass, min(16, mass.numel())).indices.tolist()
        data.append({'h': h, 'mass': mass.unsqueeze(0), 'top': top})
        print(' teacher', cid, 'loss', [round(x,3) for x in losses], 'top', top, flush=True)
        del old_items, h, mass
        gc.collect()
    return data

def train_scorer(scorer, data):
    opt = torch.optim.AdamW(scorer.parameters(), lr=LR, weight_decay=1e-4)
    curve = []
    for ep in range(EPOCHS):
        order = list(range(len(data)))
        random.Random(SEED + ep).shuffle(order)
        total = 0.0
        for i in order:
            h = data[i]['h']
            mass = data[i]['mass']
            logits = scorer(h)
            logp = F.log_softmax(logits, dim=-1)
            kl = F.kl_div(logp, mass, reduction='batchmean')
            k = min(32, logits.shape[-1])
            idx = torch.topk(mass[0], k).indices
            binary = torch.zeros_like(logits)
            binary[:, idx] = 1.0
            pos_weight = torch.tensor(max(1.0, (logits.shape[-1] - k) / max(1, k)))
            bce = F.binary_cross_entropy_with_logits(logits, binary, pos_weight=pos_weight)
            loss = kl + 0.35 * bce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().item())
        if ep % 15 == 0 or ep == EPOCHS - 1:
            rec = {'epoch': ep, 'loss': total / len(data)}
            curve.append(rec)
            print('score-train', rec, flush=True)
    return curve

@torch.no_grad()
def extend_recent(model, old_items, recent, absolute_start):
    c = dyn(clone_items(old_items))
    n = int(recent.shape[1])
    pos = torch.arange(absolute_start, absolute_start + n, dtype=torch.long)
    o = model(recent, past_key_values=c, position_ids=pos.unsqueeze(0), cache_position=pos,
              use_cache=True, return_dict=True)
    return [(k.detach().clone(), v.detach().clone()) for k, v in legacy(o.past_key_values)]

@torch.no_grad()
def select(items, score, budget):
    k = min(budget, score.numel())
    idx = torch.topk(score[0], k).indices.sort().values
    return [(x.index_select(-2, idx).contiguous(), v.index_select(-2, idx).contiguous()) for x, v in items], idx

@torch.no_grad()
def score_pack(model, pack, items):
    t = pack['t']
    seq = torch.cat([pack['last'], t[:, :-1]], dim=1)
    start = pack['cached_len']
    pos = torch.arange(start, start + seq.shape[1], dtype=torch.long)
    o = model(seq, past_key_values=dyn(clone_items(items)), position_ids=pos.unsqueeze(0), cache_position=pos,
              use_cache=False, return_dict=True)
    logits = o.logits.float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1))
    pred = logits.argmax(dim=-1)
    return {'nll': float(loss.item()), 'token_accuracy': float((pred == t).float().mean().item()),
            'exact': bool((pred == t).all().item())}

def evaluate_table(model, tok, scorer, cid):
    old, packs = prepare_table(tok, cid)
    old_items, h = prefill_old(model, old)
    pred_score = scorer(h).detach()
    teacher_mass, teacher_losses = gradient_teacher(model, old_items, packs)
    teacher_score = teacher_mass.unsqueeze(0)
    rows = []
    for pack in packs:
        full = extend_recent(model, old_items, pack['recent'], pack['old_len'])
        fm = score_pack(model, pack, full)
        recent = [(k[..., pack['old_len']:, :], v[..., pack['old_len']:, :]) for k, v in full]
        rm = score_pack(model, pack, recent)
        methods = {}
        for b in BUDGETS:
            ps, pi = select(old_items, pred_score, b)
            pm = score_pack(model, pack, extend_recent(model, ps, pack['recent'], pack['old_len']))
            ts, ti = select(old_items, teacher_score, b)
            tm = score_pack(model, pack, extend_recent(model, ts, pack['recent'], pack['old_len']))
            methods[str(b)] = {'metrics': pm, 'selected': pi.tolist(),
                               'teacher_oracle': tm, 'teacher_selected': ti.tolist()}
        rows.append({'meta': pack['meta'], 'full': fm, 'recent_only': rm, 'budgets': methods})
    return rows, teacher_losses

def summarize(rows):
    full_nll = sum(r['full']['nll'] for r in rows) / len(rows)
    recent_nll = sum(r['recent_only']['nll'] for r in rows) / len(rows)
    out = {
        'full': {'avg_nll': full_nll, 'exact_rate': sum(r['full']['exact'] for r in rows) / len(rows)},
        'recent_only': {'avg_nll': recent_nll, 'exact_rate': sum(r['recent_only']['exact'] for r in rows) / len(rows)},
    }
    denom = max(1e-9, recent_nll - full_nll)
    for b in BUDGETS:
        xs = [r['budgets'][str(b)]['metrics'] for r in rows]
        ts = [r['budgets'][str(b)]['teacher_oracle'] for r in rows]
        pn = sum(x['nll'] for x in xs) / len(xs)
        tn = sum(x['nll'] for x in ts) / len(ts)
        out[str(b)] = {
            'avg_nll': pn,
            'exact_rate': sum(x['exact'] for x in xs) / len(xs),
            'teacher_avg_nll': tn,
            'teacher_exact_rate': sum(x['exact'] for x in ts) / len(ts),
            'recovery_fraction': (recent_nll - pn) / denom,
            'teacher_recovery_fraction': (recent_nll - tn) / denom,
        }
    return out

def main():
    started = time.time()
    print('loading', MODEL_ID, flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    scorer = Scorer(int(model.config.hidden_size))
    data = build_dataset(model, tok)
    curve = train_scorer(scorer, data)
    scorer.eval()
    rows = []
    teacher_query_losses = {}
    for cid in range(20, 24):
        rr, losses = evaluate_table(model, tok, scorer, cid)
        rows.extend(rr)
        teacher_query_losses[str(cid)] = losses
        print('eval-table', cid, flush=True)
    summary = summarize(rows)
    print('summary', json.dumps(summary), flush=True)
    result = {
        'model': MODEL_ID,
        'method': 'causal scalar scorer distilled from first-order KV deletion influence |<grad,state>|',
        'budgets': BUDGETS, 'recent_tokens': RECENT, 'max_prefix': MAX_PREFIX,
        'train_tables': TRAIN_TABLES, 'scorer_params': sum(p.numel() for p in scorer.parameters()),
        'training_curve': curve, 'teacher_query_losses': teacher_query_losses,
        'heldout': rows, 'summary': summary, 'elapsed_seconds': time.time() - started,
        'guardrails': [
            'Future targets are used only to construct gradient-deletion teacher importance during training/evaluation oracle.',
            'The learned scorer sees only causal hidden states of the old prefix at inference.',
            'Evaluation uses streaming compression: selected old KV are chosen before recent/query tokens are processed.',
            'Teacher-oracle metrics test whether first-order deletion influence is itself a useful boundary target.'
        ]
    }
    OUT.write_text(json.dumps(result, indent=2))
    torch.save({'state_dict': scorer.state_dict(), 'rank': RANK}, CKPT)
    print('wrote', OUT, CKPT, flush=True)

if __name__ == '__main__':
    main()
