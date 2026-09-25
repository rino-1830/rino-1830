import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "prism-ml/Bonsai-1.7B-unpacked")
OUT = Path(os.environ.get("RESULT_PATH", "boundary-compression-lab/results/selector_boundary_results.json"))
CKPT = Path(os.environ.get("CKPT_PATH", "boundary-compression-lab/results/selector_boundary_encoder.pt"))
SLOTS = int(os.environ.get("BOUNDARY_SLOTS", "32"))
RECENT = int(os.environ.get("RECENT_TOKENS", "48"))
MAX_PREFIX = int(os.environ.get("MAX_PREFIX", "192"))
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "96"))
LR = float(os.environ.get("LR", "0.01"))
RANK = int(os.environ.get("BOUNDARY_RANK", "64"))
TRAIN_CASES = int(os.environ.get("TRAIN_CASES", "24"))
TEST_CASES = int(os.environ.get("TEST_CASES", "8"))
SEED = 11

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 1))


def make_code(i):
    left = [
        "LANTERN","ORBIT","CEDAR","MICA","NOVA","RAVEN","EMBER","LOTUS",
        "QUARTZ","MAPLE","SOLAR","VIOLET","COMET","FROST","IVORY","DELTA",
        "SABLE","CORAL","ONYX","AURORA","PINE","AMBER","FLINT","IRIS",
        "LUNAR","BIRCH","OPAL","TERRA","ZEPHYR","CINDER","JADE","KITE"
    ]
    return f"{left[i % len(left)]}-{4100 + ((i * 379) % 5800):04d}"


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
    labels = ["ALPHA","BETA","GAMMA","DELTA"]
    codes = [make_code(case_id * 4 + j) for j in range(4)]
    table = "\n".join(f"{labels[j]} = {codes[j]}" for j in range(4))
    prefix = (
        "Memorize the following identifier table exactly.\n" + table + "\n\n" +
        filler(case_id) +
        f"\nQuestion: What is the exact identifier for {labels[query_slot]}?\nAnswer:"
    )
    return prefix, " " + codes[query_slot], {"case_id": case_id, "query": labels[query_slot], "answer": codes[query_slot]}


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
    t = tok(target, add_special_tokens=False, return_tensors="pt").input_ids
    if p.shape[1] > MAX_PREFIX:
        left = MAX_PREFIX - RECENT
        p = torch.cat([p[:, :left], p[:, -RECENT:]], dim=1)
    return p, t[:, :12]


class HardSelectorBoundary(nn.Module):
    """Straight-through hard selection of original old-prefix token embeddings."""
    def __init__(self, hidden_size, slots=SLOTS, rank=RANK):
        super().__init__()
        self.slots = slots
        self.key_proj = nn.Linear(hidden_size, rank, bias=False)
        self.queries = nn.Parameter(torch.randn(slots, rank) / math.sqrt(rank))
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.8)))

    def forward(self, contextual, raw_embeddings, training_hard=True):
        k = F.normalize(self.key_proj(contextual.float()), dim=-1)
        q = F.normalize(self.queries, dim=-1)
        temp = self.log_temp.exp().clamp(0.08, 3.0)
        logits = torch.einsum("sr,bnr->bsn", q, k) / temp
        soft = logits.softmax(dim=-1)
        idx = soft.argmax(dim=-1)
        hard = F.one_hot(idx, num_classes=soft.shape[-1]).to(soft.dtype)
        weights = hard + soft - soft.detach() if training_hard else hard
        memory = torch.einsum("bsn,bnh->bsh", weights, raw_embeddings.float())

        # Preserve causal order in the student sequence. Sorting is only a routing decision;
        # gradients still flow through the selected memory vectors.
        order = idx.argsort(dim=-1)
        gather_h = order.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        memory = memory.gather(1, gather_h)
        sorted_idx = idx.gather(1, order)
        return memory, soft, sorted_idx


@torch.no_grad()
def cache_features(model, tok, prefix, target):
    p, t = encode_pair(tok, prefix, target)
    old_len = max(1, p.shape[1] - RECENT)
    old_ids = p[:, :old_len]
    recent_ids = p[:, old_len:]
    emb = model.get_input_embeddings()
    raw_old = emb(old_ids).detach().float().clone()
    recent_emb = emb(recent_ids).detach().clone()
    target_emb = emb(t).detach().clone()
    teacher = model(p, output_hidden_states=True, use_cache=False, return_dict=True)
    contextual_old = teacher.hidden_states[-1][:, :old_len, :].detach().float().clone()
    return {
        "prefix_ids": p, "target_ids": t, "old_len": old_len,
        "raw_old": raw_old, "contextual_old": contextual_old,
        "recent_ids": recent_ids, "recent_emb": recent_emb, "target_emb": target_emb,
    }


def student_batch(model, enc, item, training=True):
    mem, soft, selected = enc(item["contextual_old"], item["raw_old"], training_hard=training)
    dtype = model.get_input_embeddings().weight.dtype
    mem = mem.to(dtype)
    recent = item["recent_emb"].to(dtype)
    target_emb = item["target_emb"].to(dtype)
    x = torch.cat([mem, recent, target_emb], dim=1)

    prefix_len = item["prefix_ids"].shape[1]
    target_len = item["target_ids"].shape[1]
    selected_pos = selected.long()
    recent_pos = torch.arange(item["old_len"], prefix_len).long().unsqueeze(0)
    target_pos = torch.arange(prefix_len, prefix_len + target_len).long().unsqueeze(0)
    pos = torch.cat([selected_pos, recent_pos, target_pos], dim=1)

    labels = torch.full((1, x.shape[1]), -100, dtype=torch.long)
    labels[:, SLOTS + recent.shape[1]:] = item["target_ids"]
    out = model(inputs_embeds=x, position_ids=pos, labels=labels, use_cache=False, return_dict=True)

    gram = torch.bmm(soft, soft.transpose(1, 2))
    eye = torch.eye(SLOTS, dtype=gram.dtype, device=gram.device).unsqueeze(0)
    diversity = ((gram * (1.0 - eye)).sum() / max(1, SLOTS * (SLOTS - 1)))
    entropy = -(soft.clamp_min(1e-8) * soft.clamp_min(1e-8).log()).sum(dim=-1).mean()
    return out.loss.float(), out.logits.float(), soft, selected, labels, diversity, entropy


@torch.inference_mode()
def full_metrics(model, item):
    p, t = item["prefix_ids"], item["target_ids"]
    ids = torch.cat([p, t], dim=1)
    labels = torch.full_like(ids, -100)
    labels[:, p.shape[1]:] = t
    out = model(ids, labels=labels, use_cache=False, return_dict=True)
    start = p.shape[1] - 1
    pred = out.logits[:, start:start+t.shape[1], :].argmax(dim=-1)
    return {"nll": out.loss.float().item(),
            "token_accuracy": (pred == t).float().mean().item(),
            "exact": bool((pred == t).all().item())}


@torch.inference_mode()
def recent_metrics(model, item):
    t, recent = item["target_ids"], item["recent_ids"]
    ids = torch.cat([recent, t], dim=1)
    labels = torch.full_like(ids, -100)
    labels[:, recent.shape[1]:] = t
    prefix_len = item["prefix_ids"].shape[1]
    pos = torch.arange(item["old_len"], prefix_len + t.shape[1]).unsqueeze(0)
    out = model(ids, position_ids=pos, labels=labels, use_cache=False, return_dict=True)
    start = recent.shape[1] - 1
    pred = out.logits[:, start:start+t.shape[1], :].argmax(dim=-1)
    return {"nll": out.loss.float().item(),
            "token_accuracy": (pred == t).float().mean().item(),
            "exact": bool((pred == t).all().item())}


@torch.inference_mode()
def selector_metrics(model, enc, item):
    loss, logits, soft, selected, _, diversity, entropy = student_batch(model, enc, item, training=False)
    target_start = SLOTS + item["recent_ids"].shape[1]
    n = item["target_ids"].shape[1]
    pred = logits[:, target_start-1:target_start-1+n, :].argmax(dim=-1)
    t = item["target_ids"]
    sel = selected[0].tolist()
    return {
        "nll": loss.item(),
        "token_accuracy": (pred == t).float().mean().item(),
        "exact": bool((pred == t).all().item()),
        "selected_old_positions": sel,
        "unique_selected_positions": len(set(sel)),
        "entropy": entropy.item(),
        "diversity": diversity.item(),
    }


def main():
    started = time.time()
    print("loading", MODEL_ID, flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hidden = int(model.config.hidden_size)
    enc = HardSelectorBoundary(hidden)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=1e-4)

    specs = [(i, q) for i in range(20) for q in range(4)]
    random.shuffle(specs)
    specs = specs[:TRAIN_CASES]
    test_specs = [(40+i, i % 4) for i in range(TEST_CASES)]

    print("caching frozen-model features", flush=True)
    train = []
    for i, q in specs:
        p, t, meta = make_case(i, q)
        x = cache_features(model, tok, p, t)
        x["meta"] = meta
        train.append(x)
        print(" train", meta, "old", x["old_len"], "target", x["target_ids"].shape[1], flush=True)
    test = []
    for i, q in test_specs:
        p, t, meta = make_case(i, q)
        x = cache_features(model, tok, p, t)
        x["meta"] = meta
        test.append(x)
        print(" test", meta, "old", x["old_len"], "target", x["target_ids"].shape[1], flush=True)

    curve = []
    enc.train()
    for step in range(TRAIN_STEPS):
        item = train[step % len(train)]
        opt.zero_grad(set_to_none=True)
        loss, _, soft, selected, _, diversity, entropy = student_batch(model, enc, item, training=True)
        # Selection should become sharp and slots should not collapse onto the same old token.
        objective = loss + 0.20 * diversity + 0.001 * entropy
        objective.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        if step % 8 == 0 or step == TRAIN_STEPS - 1:
            uniq = len(set(selected[0].tolist()))
            rec = {"step": step, "nll": loss.item(), "diversity": diversity.item(),
                   "entropy": entropy.item(), "temperature": enc.log_temp.exp().item(),
                   "unique_selected": uniq}
            curve.append(rec)
            print("step", rec, flush=True)

    enc.eval()
    evals = []
    for item in test:
        fm, rm, sm = full_metrics(model, item), recent_metrics(model, item), selector_metrics(model, enc, item)
        row = {"meta": item["meta"], "prefix_tokens": int(item["prefix_ids"].shape[1]),
               "old_tokens": int(item["old_len"]), "recent_tokens": int(item["recent_ids"].shape[1]),
               "target_tokens": int(item["target_ids"].shape[1]),
               "full": fm, "recent_only": rm, "selector_boundary": sm}
        evals.append(row)
        print("eval", item["meta"], "full", fm, "recent", rm,
              "selector", {k:v for k,v in sm.items() if k != "selected_old_positions"}, flush=True)

    nprefix, ntarget = natural_case()
    natural = cache_features(model, tok, nprefix, ntarget)
    natural_eval = {"prefix_tokens": int(natural["prefix_ids"].shape[1]),
                    "old_tokens": int(natural["old_len"]),
                    "full": full_metrics(model, natural),
                    "recent_only": recent_metrics(model, natural),
                    "selector_boundary": selector_metrics(model, enc, natural)}

    def rate(path):
        return sum(x[path]["exact"] for x in evals) / len(evals)
    def avg(path, field="nll"):
        return sum(x[path][field] for x in evals) / len(evals)

    summary = {
        "full_exact_rate": rate("full"),
        "recent_exact_rate": rate("recent_only"),
        "selector_exact_rate": rate("selector_boundary"),
        "full_avg_nll": avg("full"),
        "recent_avg_nll": avg("recent_only"),
        "selector_avg_nll": avg("selector_boundary"),
        "selector_delta_nll_vs_full": avg("selector_boundary") - avg("full"),
        "selector_delta_nll_vs_recent": avg("selector_boundary") - avg("recent_only"),
        "avg_unique_selected": avg("selector_boundary", "unique_selected_positions"),
        "old_compression_ratio": evals[0]["old_tokens"] / SLOTS,
        "total_prefix_compression_ratio": evals[0]["prefix_tokens"] / (SLOTS + RECENT),
    }
    result = {
        "model": MODEL_ID, "method": "straight_through_hard_original_token_selector",
        "boundary_slots": SLOTS, "recent_tokens": RECENT, "max_prefix": MAX_PREFIX,
        "train_steps": TRAIN_STEPS, "train_cases": TRAIN_CASES, "test_cases": TEST_CASES,
        "rank": RANK, "training_curve": curve, "heldout_retrieval": evals,
        "heldout_summary": summary, "natural_check": natural_eval,
        "parameter_count": sum(p.numel() for p in enc.parameters()),
        "elapsed_seconds": time.time() - started,
        "guardrails": [
            "Bonsai-1.7B remains frozen; only the selector is trained.",
            "Held-out case IDs do not appear in training.",
            "Forward selection is hard one-hot; soft probabilities are used only for straight-through gradients.",
            "This controlled retrieval test is not yet a general benchmark."
        ]
    }
    OUT.write_text(json.dumps(result, indent=2))
    torch.save({"state_dict": enc.state_dict(), "config": {"hidden_size": hidden, "slots": SLOTS, "rank": RANK}}, CKPT)
    print("summary", summary, flush=True)
    print("natural", {k:v for k,v in natural_eval.items() if k != "selector_boundary"},
          {k:v for k,v in natural_eval["selector_boundary"].items() if k != "selected_old_positions"}, flush=True)
    print("wrote", OUT, CKPT, flush=True)

if __name__ == "__main__":
    main()
