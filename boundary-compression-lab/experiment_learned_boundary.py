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
OUT = Path(os.environ.get("RESULT_PATH", "boundary-compression-lab/results/learned_boundary_results.json"))
CKPT = Path(os.environ.get("CKPT_PATH", "boundary-compression-lab/results/learned_boundary_encoder.pt"))
SLOTS = int(os.environ.get("BOUNDARY_SLOTS", "16"))
RECENT = int(os.environ.get("RECENT_TOKENS", "48"))
MAX_PREFIX = int(os.environ.get("MAX_PREFIX", "192"))
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "48"))
LR = float(os.environ.get("LR", "0.015"))
RANK = int(os.environ.get("BOUNDARY_RANK", "64"))
SEED = 7

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 1))


def make_code(i):
    left = ["LANTERN", "ORBIT", "CEDAR", "MICA", "NOVA", "RAVEN", "EMBER", "LOTUS",
            "QUARTZ", "MAPLE", "SOLAR", "VIOLET", "COMET", "FROST", "IVORY", "DELTA",
            "SABLE", "CORAL", "ONYX", "AURORA", "PINE", "AMBER", "FLINT", "IRIS",
            "LUNAR", "BIRCH", "OPAL", "TERRA", "ZEPHYR", "CINDER", "JADE", "KITE"]
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
    labels = ["ALPHA", "BETA", "GAMMA", "DELTA"]
    codes = [make_code(case_id * 4 + j) for j in range(4)]
    table = "\n".join(f"{labels[j]} = {codes[j]}" for j in range(4))
    prefix = (
        "Memorize the following identifier table exactly.\n"
        + table
        + "\n\n"
        + filler(case_id)
        + f"\nQuestion: What is the exact identifier for {labels[query_slot]}?\nAnswer:"
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
        # Keep the memory table at the beginning and the question at the end.
        left = MAX_PREFIX - RECENT
        p = torch.cat([p[:, :left], p[:, -RECENT:]], dim=1)
    return p, t[:, :12]


class BoundaryEncoder(nn.Module):
    def __init__(self, hidden_size, slots=SLOTS, rank=RANK):
        super().__init__()
        self.slots = slots
        self.rank = rank
        self.key_proj = nn.Linear(hidden_size, rank, bias=False)
        self.queries = nn.Parameter(torch.randn(slots, rank) / math.sqrt(rank))
        self.adapter_down = nn.Linear(hidden_size, rank, bias=False)
        self.adapter_up = nn.Linear(rank, hidden_size, bias=False)
        self.slot_bias = nn.Parameter(torch.zeros(slots, hidden_size))
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.7)))

    def forward(self, contextual, raw_embeddings):
        # contextual/raw_embeddings: [1, old_len, hidden]
        k = F.normalize(self.key_proj(contextual.float()), dim=-1)
        q = F.normalize(self.queries, dim=-1)
        temp = self.log_temp.exp().clamp(0.08, 4.0)
        score = torch.einsum("sr,bnr->bsn", q, k) / temp
        w = score.softmax(dim=-1)
        pooled = torch.einsum("bsn,bnh->bsh", w, raw_embeddings.float())
        delta = self.adapter_up(torch.tanh(self.adapter_down(pooled)))
        memory = pooled + 0.10 * delta + 0.02 * self.slot_bias.unsqueeze(0)
        return memory, w


@torch.no_grad()
def cache_features(model, tok, prefix, target):
    p, t = encode_pair(tok, prefix, target)
    old_len = max(1, p.shape[1] - RECENT)
    old_ids = p[:, :old_len]
    recent_ids = p[:, old_len:]
    emb = model.get_input_embeddings()
    raw_old = emb(old_ids).detach().float().clone()
    recent_emb = emb(recent_ids).detach()
    target_emb = emb(t).detach()
    teacher = model(p, output_hidden_states=True, use_cache=False, return_dict=True)
    contextual_old = teacher.hidden_states[-1][:, :old_len, :].detach().float().clone()
    return {
        "prefix_ids": p,
        "target_ids": t,
        "old_len": old_len,
        "raw_old": raw_old,
        "contextual_old": contextual_old,
        "recent_ids": recent_ids,
        "recent_emb": recent_emb,
        "target_emb": target_emb,
    }


def student_batch(model, enc, item):
    mem, weights = enc(item["contextual_old"], item["raw_old"])
    dtype = model.get_input_embeddings().weight.dtype
    mem = mem.to(dtype)
    recent = item["recent_emb"].to(dtype)
    target_emb = item["target_emb"].to(dtype)
    x = torch.cat([mem, recent, target_emb], dim=1)

    # Memory slots represent the distant prefix; recent tokens retain their original positions.
    old_len = item["old_len"]
    prefix_len = item["prefix_ids"].shape[1]
    target_len = item["target_ids"].shape[1]
    mem_pos = torch.linspace(0, max(0, old_len - 1), SLOTS).round().long().unsqueeze(0)
    recent_pos = torch.arange(old_len, prefix_len).long().unsqueeze(0)
    target_pos = torch.arange(prefix_len, prefix_len + target_len).long().unsqueeze(0)
    pos = torch.cat([mem_pos, recent_pos, target_pos], dim=1)

    labels = torch.full((1, x.shape[1]), -100, dtype=torch.long)
    labels[:, SLOTS + recent.shape[1]:] = item["target_ids"]
    out = model(inputs_embeds=x, position_ids=pos, labels=labels, use_cache=False, return_dict=True)
    return out.loss.float(), out.logits.float(), weights, labels


@torch.inference_mode()
def full_metrics(model, item):
    p = item["prefix_ids"]
    t = item["target_ids"]
    ids = torch.cat([p, t], dim=1)
    labels = torch.full_like(ids, -100)
    labels[:, p.shape[1]:] = t
    out = model(ids, labels=labels, use_cache=False, return_dict=True)
    start = p.shape[1] - 1
    pred = out.logits[:, start:start+t.shape[1], :].argmax(dim=-1)
    acc = (pred == t).float().mean().item()
    exact = bool((pred == t).all().item())
    return {"nll": out.loss.float().item(), "token_accuracy": acc, "exact": exact}


@torch.inference_mode()
def recent_metrics(model, item):
    t = item["target_ids"]
    recent = item["recent_ids"]
    ids = torch.cat([recent, t], dim=1)
    labels = torch.full_like(ids, -100)
    labels[:, recent.shape[1]:] = t
    # Preserve original absolute positions.
    prefix_len = item["prefix_ids"].shape[1]
    recent_start = item["old_len"]
    pos = torch.arange(recent_start, prefix_len + t.shape[1]).unsqueeze(0)
    out = model(ids, position_ids=pos, labels=labels, use_cache=False, return_dict=True)
    start = recent.shape[1] - 1
    pred = out.logits[:, start:start+t.shape[1], :].argmax(dim=-1)
    return {"nll": out.loss.float().item(),
            "token_accuracy": (pred == t).float().mean().item(),
            "exact": bool((pred == t).all().item())}


@torch.inference_mode()
def learned_metrics(model, enc, item):
    loss, logits, w, labels = student_batch(model, enc, item)
    target_start = SLOTS + item["recent_ids"].shape[1]
    # Causal prediction for label at target_start is logits[target_start-1].
    n = item["target_ids"].shape[1]
    pred = logits[:, target_start-1:target_start-1+n, :].argmax(dim=-1)
    t = item["target_ids"]
    top_positions = torch.topk(w[0], k=min(3, w.shape[-1]), dim=-1).indices.tolist()
    return {
        "nll": loss.item(),
        "token_accuracy": (pred == t).float().mean().item(),
        "exact": bool((pred == t).all().item()),
        "top_old_positions_per_slot": top_positions,
    }


def main():
    started = time.time()
    print("loading", MODEL_ID, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hidden = int(model.config.hidden_size)
    enc = BoundaryEncoder(hidden)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=1e-4)

    # Held-out case IDs are disjoint, so the encoder cannot memorize their codes.
    train_specs = [(i, q) for i in range(12) for q in range(4)]
    random.shuffle(train_specs)
    train_specs = train_specs[:16]
    test_specs = [(20+i, i % 4) for i in range(4)]

    print("caching frozen-model features", flush=True)
    train = []
    for i, q in train_specs:
        p, t, meta = make_case(i, q)
        x = cache_features(model, tok, p, t)
        x["meta"] = meta
        train.append(x)
        print(" train", meta, "prefix", x["prefix_ids"].shape[1], "target", x["target_ids"].shape[1], flush=True)

    test = []
    for i, q in test_specs:
        p, t, meta = make_case(i, q)
        x = cache_features(model, tok, p, t)
        x["meta"] = meta
        test.append(x)
        print(" test", meta, "prefix", x["prefix_ids"].shape[1], "target", x["target_ids"].shape[1], flush=True)

    training_curve = []
    enc.train()
    for step in range(TRAIN_STEPS):
        item = train[step % len(train)]
        opt.zero_grad(set_to_none=True)
        loss, _, w, _ = student_batch(model, enc, item)
        # Encourage each slot to choose a compact part of the prefix without forcing a hard one-hot.
        entropy = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(dim=-1).mean()
        objective = loss + 0.002 * entropy
        objective.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        if step % 4 == 0 or step == TRAIN_STEPS - 1:
            rec = {"step": step, "nll": loss.item(), "entropy": entropy.item(),
                   "temperature": enc.log_temp.exp().item()}
            training_curve.append(rec)
            print("step", rec, flush=True)

    enc.eval()
    evaluations = []
    for item in test:
        fm = full_metrics(model, item)
        rm = recent_metrics(model, item)
        lm = learned_metrics(model, enc, item)
        evaluations.append({"meta": item["meta"], "prefix_tokens": int(item["prefix_ids"].shape[1]),
                            "old_tokens": int(item["old_len"]), "recent_tokens": int(item["recent_ids"].shape[1]),
                            "target_tokens": int(item["target_ids"].shape[1]),
                            "full": fm, "recent_only": rm, "learned_boundary": lm})
        print("eval", item["meta"], "full", fm, "recent", rm, "learned", {k:v for k,v in lm.items() if k != "top_old_positions_per_slot"}, flush=True)

    # One out-of-domain natural-language check.
    nprefix, ntarget = natural_case()
    natural = cache_features(model, tok, nprefix, ntarget)
    natural_eval = {"prefix_tokens": int(natural["prefix_ids"].shape[1]),
                    "old_tokens": int(natural["old_len"]),
                    "full": full_metrics(model, natural),
                    "recent_only": recent_metrics(model, natural),
                    "learned_boundary": learned_metrics(model, enc, natural)}
    print("natural", {k:v for k,v in natural_eval.items() if k != "learned_boundary"},
          {k:v for k,v in natural_eval["learned_boundary"].items() if k != "top_old_positions_per_slot"}, flush=True)

    exact_full = sum(x["full"]["exact"] for x in evaluations) / len(evaluations)
    exact_recent = sum(x["recent_only"]["exact"] for x in evaluations) / len(evaluations)
    exact_learned = sum(x["learned_boundary"]["exact"] for x in evaluations) / len(evaluations)
    avg_full = sum(x["full"]["nll"] for x in evaluations) / len(evaluations)
    avg_recent = sum(x["recent_only"]["nll"] for x in evaluations) / len(evaluations)
    avg_learned = sum(x["learned_boundary"]["nll"] for x in evaluations) / len(evaluations)

    result = {
        "model": MODEL_ID,
        "boundary_slots": SLOTS,
        "recent_tokens": RECENT,
        "total_student_slots_before_targets": SLOTS + RECENT,
        "max_prefix": MAX_PREFIX,
        "train_steps": TRAIN_STEPS,
        "rank": RANK,
        "train_cases": [x["meta"] for x in train],
        "training_curve": training_curve,
        "heldout_retrieval": evaluations,
        "heldout_summary": {
            "full_exact_rate": exact_full,
            "recent_exact_rate": exact_recent,
            "learned_exact_rate": exact_learned,
            "full_avg_nll": avg_full,
            "recent_avg_nll": avg_recent,
            "learned_avg_nll": avg_learned,
            "learned_delta_nll_vs_full": avg_learned - avg_full,
        },
        "natural_check": natural_eval,
        "parameter_count": sum(p.numel() for p in enc.parameters()),
        "elapsed_seconds": time.time() - started,
        "interpretation_guardrails": [
            "The Bonsai model is frozen; only the small boundary encoder is trained.",
            "Held-out retrieval identifiers use disjoint case IDs from training.",
            "The encoder pools frozen contextual states to decide which raw token embeddings should form soft memory tokens.",
            "This is a proof-of-concept for learned boundary compression, not yet a general language-model benchmark."
        ]
    }
    OUT.write_text(json.dumps(result, indent=2))
    torch.save({"state_dict": enc.state_dict(), "config": {"hidden_size": hidden, "slots": SLOTS, "rank": RANK}}, CKPT)
    print("summary", result["heldout_summary"], flush=True)
    print("wrote", OUT, CKPT, flush=True)


if __name__ == "__main__":
    main()
