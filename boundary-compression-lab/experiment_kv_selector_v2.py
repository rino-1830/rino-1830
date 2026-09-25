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

MODEL_ID = os.environ.get("MODEL_ID", "prism-ml/Bonsai-1.7B-unpacked")
OUT = Path(os.environ.get("RESULT_PATH", "boundary-compression-lab/results/kv_selector_v2.json"))
CKPT = Path(os.environ.get("CKPT_PATH", "boundary-compression-lab/results/kv_selector_v2.pt"))
SLOTS = int(os.environ.get("BOUNDARY_SLOTS", "32"))
RECENT = int(os.environ.get("RECENT_TOKENS", "48"))
MAX_PREFIX = int(os.environ.get("MAX_PREFIX", "192"))
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "96"))
LR = float(os.environ.get("LR", "0.01"))
RANK = int(os.environ.get("BOUNDARY_RANK", "64"))
THREADS = int(os.environ.get("TORCH_THREADS", "4"))
SEED = int(os.environ.get("SEED", "13"))

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_num_threads(min(THREADS, os.cpu_count() or 1))

LABELS = ["ALPHA", "BETA", "GAMMA", "DELTA"]
WORDS = ["LANTERN","ORBIT","CEDAR","MICA","NOVA","RAVEN","EMBER","LOTUS",
         "QUARTZ","MAPLE","SOLAR","VIOLET","COMET","FROST","IVORY","DELTA",
         "SABLE","CORAL","ONYX","AURORA","PINE","AMBER","FLINT","IRIS",
         "LUNAR","BIRCH","OPAL","TERRA","ZEPHYR","CINDER","JADE","KITE"]

def make_code(i):
    return f"{WORDS[i % len(WORDS)]}-{4100 + ((i * 379) % 5800):04d}"

def filler(seed, repeats=10):
    variants = [
        "A field notebook records routine observations about tools, roads, materials, weather, and schedules. ",
        "The maintenance log lists ordinary checks, measurements, replacement parts, and inspection notes. ",
        "Unrelated notes describe storage rooms, work benches, cables, labels, and daily operating procedures. ",
        "The report contains mundane details that should not replace the identifiers stored in the table. ",
    ]
    r = random.Random(seed)
    return "".join(r.choice(variants) for _ in range(repeats))

def make_case(case_id, query_slot):
    codes = [make_code(case_id * 4 + j) for j in range(4)]
    table = "\n".join(f"{LABELS[j]} = {codes[j]}" for j in range(4))
    prefix = (
        "Memorize the following identifier table exactly.\n" + table + "\n\n"
        + filler(case_id)
        + f"\nQuestion: What is the exact identifier for {LABELS[query_slot]}?\nAnswer:"
    )
    return prefix, " " + codes[query_slot], {
        "case_id": case_id, "query": LABELS[query_slot], "answer": codes[query_slot]
    }

def natural_case():
    prefix = (
        "A transformer language model predicts the next token from a sequence of earlier tokens. "
        "During autoregressive inference, attention keys and values from earlier positions are cached "
        "so they do not need to be recomputed. As the context grows, this cache can become a large "
        "fraction of total memory use. One possible strategy is to preserve only the information from "
        "the distant prefix that is useful for predicting the continuation. In that setting, the main "
        "question is whether a small boundary state can replace most of the old cache without changing "
    )
    return prefix, "the model prediction very much."

def encode_pair(tok, prefix, target):
    p = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids
    t = tok(target, add_special_tokens=False, return_tensors="pt").input_ids[:, :12]
    if p.shape[1] > MAX_PREFIX:
        left = MAX_PREFIX - RECENT
        p = torch.cat([p[:, :left], p[:, -RECENT:]], dim=1)
    return p, t

def legacy(cache):
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(cache)

def dyn(items):
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(items))
    c = DynamicCache()
    for i, (k, v) in enumerate(items):
        c.update(k, v, i)
    return c

def clone_items(items):
    return [(k.clone(), v.clone()) for k, v in items]

class KVSelector(nn.Module):
    def __init__(self, hidden, slots=SLOTS, rank=RANK):
        super().__init__()
        self.key_proj = nn.Linear(hidden, rank, bias=False)
        self.queries = nn.Parameter(torch.randn(slots, rank) / math.sqrt(rank))
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.7)))

    def soft_weights(self, contextual):
        k = F.normalize(self.key_proj(contextual.float()), dim=-1)
        q = F.normalize(self.queries, dim=-1)
        temp = self.log_temp.exp().clamp(0.08, 4.0)
        return (torch.einsum("sr,bnr->bsn", q, k) / temp).softmax(dim=-1)

    def forward(self, contextual):
        soft = self.soft_weights(contextual)[0]
        chosen = []
        used = torch.zeros(soft.shape[-1], dtype=torch.bool, device=soft.device)
        detached = soft.detach()
        for s in range(soft.shape[0]):
            idx = int(detached[s].masked_fill(used, -1.0).argmax().item())
            chosen.append(idx)
            used[idx] = True
        chosen_t = torch.tensor(chosen, device=soft.device)
        order = torch.argsort(chosen_t)
        chosen_t = chosen_t.index_select(0, order)
        hard = F.one_hot(chosen_t, num_classes=soft.shape[-1]).to(soft.dtype)
        soft_ord = soft.index_select(0, order)
        st = hard + soft_ord - soft_ord.detach()
        return st, soft_ord, chosen_t

def prepare_case(tok, case_id):
    packs = []
    common_old = None
    common_old_len = None
    for q in range(4):
        prefix, target, meta = make_case(case_id, q)
        p, t = encode_pair(tok, prefix, target)
        cached = p[:, :-1]
        old_len = max(1, cached.shape[1] - RECENT)
        old_ids = cached[:, :old_len]
        recent_ids = cached[:, old_len:]
        if common_old is None:
            common_old = old_ids
            common_old_len = old_len
        else:
            if old_len != common_old_len or not torch.equal(old_ids, common_old):
                raise RuntimeError("query leaked into old prefix; cannot share cache safely")
        packs.append({
            "prefix_ids": p,
            "target_ids": t,
            "recent_ids": recent_ids,
            "last_prefix": p[:, -1:],
            "cached_len": int(cached.shape[1]),
            "old_len": int(old_len),
            "meta": meta,
        })
    return common_old, packs

@torch.no_grad()
def prefill_old(model, old_ids):
    out = model(old_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    items = [(k.detach().clone(), v.detach().clone()) for k, v in legacy(out.past_key_values)]
    contextual = out.hidden_states[-1].detach().float().clone()
    return items, contextual

@torch.no_grad()
def extend_recent(model, base_items, recent_ids, absolute_start):
    cache = dyn(clone_items(base_items))
    n = int(recent_ids.shape[1])
    pos = torch.arange(absolute_start, absolute_start + n, dtype=torch.long)
    out = model(
        recent_ids,
        past_key_values=cache,
        position_ids=pos.unsqueeze(0),
        cache_position=pos,
        use_cache=True,
        return_dict=True,
    )
    return [(k.detach().clone(), v.detach().clone()) for k, v in legacy(out.past_key_values)]

def select_old_items(old_items, st):
    out = []
    for k, v in old_items:
        ck = torch.einsum("sn,bhnd->bhsd", st, k)
        cv = torch.einsum("sn,bhnd->bhsd", st, v)
        out.append((ck, cv))
    return out

def select_posthoc_items(full_items, old_len, st):
    out = []
    for k, v in full_items:
        oldk, oldv = k[..., :old_len, :], v[..., :old_len, :]
        recentk, recentv = k[..., old_len:, :], v[..., old_len:, :]
        ck = torch.einsum("sn,bhnd->bhsd", st, oldk)
        cv = torch.einsum("sn,bhnd->bhsd", st, oldv)
        out.append((torch.cat([ck, recentk], dim=-2), torch.cat([cv, recentv], dim=-2)))
    return out

def recent_only_items(full_items, old_len):
    return [(k[..., old_len:, :].clone(), v[..., old_len:, :].clone()) for k, v in full_items]

def score_items(model, pack, items):
    t = pack["target_ids"]
    seq = torch.cat([pack["last_prefix"], t[:, :-1]], dim=1)
    start = pack["cached_len"]
    pos = torch.arange(start, start + seq.shape[1], dtype=torch.long)
    out = model(
        seq,
        past_key_values=dyn(items),
        position_ids=pos.unsqueeze(0),
        cache_position=pos,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits.float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1))
    pred = logits.argmax(dim=-1)
    metrics = {
        "nll": float(loss.detach().item()),
        "token_accuracy": float((pred == t).float().mean().item()),
        "exact": bool((pred == t).all().item()),
    }
    return loss, metrics

def regularizers(soft):
    z = F.normalize(soft, p=2, dim=-1)
    gram = z @ z.T
    eye = torch.eye(gram.shape[0], device=gram.device)
    diversity = (gram - eye).pow(2).mean()
    entropy = -(soft.clamp_min(1e-8) * soft.clamp_min(1e-8).log()).sum(dim=-1).mean()
    return diversity, entropy

def train_one(model, selector, opt, old_items, contextual, pack):
    with torch.no_grad():
        full_items = extend_recent(model, old_items, pack["recent_ids"], pack["old_len"])
    st, soft, pos = selector(contextual)
    selected = select_posthoc_items(full_items, pack["old_len"], st)
    loss, metrics = score_items(model, pack, selected)
    diversity, entropy = regularizers(soft)
    objective = loss + 0.08 * diversity + 0.003 * entropy
    opt.zero_grad(set_to_none=True)
    objective.backward()
    torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
    opt.step()
    return {
        "nll": metrics["nll"],
        "diversity_loss": float(diversity.detach().item()),
        "entropy": float(entropy.detach().item()),
        "temperature": float(selector.log_temp.exp().detach().item()),
        "selected": pos.detach().tolist(),
    }

@torch.no_grad()
def evaluate_pack(model, selector, old_items, contextual, pack):
    full_items = extend_recent(model, old_items, pack["recent_ids"], pack["old_len"])
    _, full = score_items(model, pack, clone_items(full_items))
    _, recent = score_items(model, pack, recent_only_items(full_items, pack["old_len"]))

    st, soft, pos = selector(contextual)
    _, posthoc = score_items(model, pack, select_posthoc_items(full_items, pack["old_len"], st))
    posthoc["selected_old_positions"] = pos.tolist()

    streaming = None
    streaming_error = None
    try:
        selected_old = select_old_items(old_items, st)
        streamed = extend_recent(model, selected_old, pack["recent_ids"], pack["old_len"])
        _, streaming = score_items(model, pack, streamed)
        streaming["selected_old_positions"] = pos.tolist()
    except Exception as e:
        streaming_error = repr(e)

    return {
        "full": full,
        "recent_only": recent,
        "selected_posthoc": posthoc,
        "selected_streaming": streaming,
        "streaming_error": streaming_error,
        "mean_selector_entropy": float((-(soft.clamp_min(1e-8) * soft.clamp_min(1e-8).log()).sum(dim=-1).mean()).item()),
    }

def natural_pack(tok):
    prefix, target = natural_case()
    p, t = encode_pair(tok, prefix, target)
    cached = p[:, :-1]
    old_len = max(1, cached.shape[1] - RECENT)
    return cached[:, :old_len], {
        "prefix_ids": p,
        "target_ids": t,
        "recent_ids": cached[:, old_len:],
        "last_prefix": p[:, -1:],
        "cached_len": int(cached.shape[1]),
        "old_len": int(old_len),
        "meta": {"case_id": "natural"},
    }

def summarize(evals):
    def vals(method, field):
        return [float(e[method][field]) for e in evals if e.get(method) is not None]
    out = {}
    for method in ["full", "recent_only", "selected_posthoc", "selected_streaming"]:
        xs = vals(method, "nll")
        if xs:
            out[method + "_avg_nll"] = sum(xs) / len(xs)
            ex = vals(method, "exact")
            out[method + "_exact_rate"] = sum(ex) / len(ex)
    if "selected_posthoc_avg_nll" in out:
        out["posthoc_delta_nll_vs_full"] = out["selected_posthoc_avg_nll"] - out["full_avg_nll"]
        out["posthoc_improvement_vs_recent"] = out["recent_only_avg_nll"] - out["selected_posthoc_avg_nll"]
    if "selected_streaming_avg_nll" in out:
        out["streaming_delta_nll_vs_full"] = out["selected_streaming_avg_nll"] - out["full_avg_nll"]
        out["streaming_improvement_vs_recent"] = out["recent_only_avg_nll"] - out["selected_streaming_avg_nll"]
    return out

def main():
    started = time.time()
    print("loading", MODEL_ID, "slots", SLOTS, "threads", THREADS, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    selector = KVSelector(int(model.config.hidden_size))
    opt = torch.optim.AdamW(selector.parameters(), lr=LR, weight_decay=1e-4)

    curve = []
    step = 0
    epoch = 0
    selector.train()
    while step < TRAIN_STEPS:
        case_order = list(range(12))
        random.Random(SEED + epoch).shuffle(case_order)
        for case_id in case_order:
            if step >= TRAIN_STEPS:
                break
            old_ids, packs = prepare_case(tok, case_id)
            old_items, contextual = prefill_old(model, old_ids)
            q_order = list(range(4))
            random.Random(SEED * 100 + epoch * 17 + case_id).shuffle(q_order)
            for qi in q_order:
                if step >= TRAIN_STEPS:
                    break
                rec = train_one(model, selector, opt, old_items, contextual, packs[qi])
                if step % 8 == 0 or step == TRAIN_STEPS - 1:
                    rec = {"step": step, "case_id": case_id, "query": packs[qi]["meta"]["query"], **rec}
                    curve.append(rec)
                    print("step", rec, flush=True)
                step += 1
            del old_items, contextual
        epoch += 1

    selector.eval()
    evals = []
    for case_id in range(20, 24):
        old_ids, packs = prepare_case(tok, case_id)
        old_items, contextual = prefill_old(model, old_ids)
        for pack in packs:
            m = evaluate_pack(model, selector, old_items, contextual, pack)
            row = {
                "meta": pack["meta"],
                "prefix_tokens": int(pack["prefix_ids"].shape[1]),
                "old_tokens": pack["old_len"],
                "recent_tokens": int(pack["recent_ids"].shape[1]),
                **m,
            }
            evals.append(row)
            print(
                "eval", pack["meta"],
                "full", row["full"]["exact"], round(row["full"]["nll"], 3),
                "recent", row["recent_only"]["exact"], round(row["recent_only"]["nll"], 3),
                "posthoc", row["selected_posthoc"]["exact"], round(row["selected_posthoc"]["nll"], 3),
                "stream", None if row["selected_streaming"] is None else row["selected_streaming"]["exact"],
                None if row["selected_streaming"] is None else round(row["selected_streaming"]["nll"], 3),
                flush=True,
            )
        del old_items, contextual

    n_old, n_pack = natural_pack(tok)
    n_old_items, n_contextual = prefill_old(model, n_old)
    natural = evaluate_pack(model, selector, n_old_items, n_contextual, n_pack)

    summary = summarize(evals)
    result = {
        "model": MODEL_ID,
        "method": "memory-efficient learned selector over actual KV positions",
        "boundary_slots": SLOTS,
        "recent_tokens": RECENT,
        "retained_prefix_slots": SLOTS + RECENT,
        "max_prefix": MAX_PREFIX,
        "old_region_compression_ratio": (MAX_PREFIX - RECENT) / SLOTS,
        "prefix_slot_compression_ratio": MAX_PREFIX / (SLOTS + RECENT),
        "train_steps": TRAIN_STEPS,
        "rank": RANK,
        "selector_parameters": sum(p.numel() for p in selector.parameters()),
        "training_curve": curve,
        "heldout_retrieval": evals,
        "heldout_summary": summary,
        "natural_check": natural,
        "elapsed_seconds": time.time() - started,
        "guardrails": [
            "Bonsai weights are frozen; only the selector is trained.",
            "Training processes one table at a time and does not retain all KV caches in RAM.",
            "All four questions for each training table share the same causal old-prefix representation.",
            "Selected_posthoc compresses after the recent window was computed with full old KV.",
            "Selected_streaming compresses old KV before processing the recent window and is the more deployment-realistic metric.",
            "Held-out identifiers use disjoint case IDs.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2))
    torch.save({"state_dict": selector.state_dict(), "config": {
        "hidden": int(model.config.hidden_size), "slots": SLOTS, "rank": RANK
    }}, CKPT)
    print("summary", summary, flush=True)
    print("natural", natural, flush=True)
    print("wrote", OUT, CKPT, flush=True)

if __name__ == "__main__":
    main()
