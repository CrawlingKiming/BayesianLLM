import argparse, json, math, os, random, re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset

# -----------------------------
# Helpers
# -----------------------------

def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # keep only first number-like token
        m = re.search(r"-?\d+(\.\d+)?", x)
        return float(m.group(0)) if m else None
    return None

def extract_attr_rating(completion: Dict[str, Any], attr: str) -> Optional[float]:
    """
    UltraFeedback stores per-attribute ratings under completion['annotations'][attr].
    It can be:
      - a dict with key 'Rating' (string or number),
      - OR a list of dicts with 'Rating' fields (take mean),
      - OR occasionally missing.
    """
    ann = completion.get("annotations", {}).get(attr)
    if ann is None:
        return None

    if isinstance(ann, dict):
        return _to_float(ann.get("Rating") or ann.get("rating"))

    if isinstance(ann, list):
        vals = []
        for a in ann:
            v = _to_float(a.get("Rating") or a.get("rating"))
            if v is not None:
                vals.append(v)
        if vals:
            return sum(vals) / len(vals)

    return None

def choose_pair_by_attr(
    completions: List[Dict[str, Any]],
    attr: str,
    min_gap: float = 0.0,
) -> Optional[Tuple[Dict, Dict, float, float]]:
    """
    Pick chosen = argmax rating(attr), rejected = argmin rating(attr).
    Break ties randomly. Enforce rating gap >= min_gap if provided.
    Returns (chosen_completion, rejected_completion, chosen_score, rejected_score)
    or None if cannot form a valid pair.
    """
    scored = []
    for c in completions:
        s = extract_attr_rating(c, attr)
        if s is not None and isinstance(c.get("response"), str) and c["response"].strip():
            scored.append((s, c))
    if len(scored) < 2:
        return None

    # Stable random tie-break: shuffle then sort
    random.shuffle(scored)
    scored.sort(key=lambda t: t[0])  # ascending by score

    rejected_score, rejected = scored[0]
    chosen_score, chosen = scored[-1]

    if min_gap is not None and (chosen_score - rejected_score) < min_gap:
        return None

    return chosen, rejected, float(chosen_score), float(rejected_score)

def split_by_prompt_id(ids: List[str], test_size: float, seed: int):
    random.Random(seed).shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_size)))
    test_ids = set(ids[:n_test])
    train_ids = set(ids[n_test:])
    return train_ids, test_ids

# -----------------------------
# Main builder
# -----------------------------

def build_uf_p2(out_dir: str, test_size: float = 0.1, seed: int = 7, min_gap: float = 0.0):
    random.seed(seed)

    ds = load_dataset("openbmb/UltraFeedback", split="train")
    os.makedirs(out_dir, exist_ok=True)

    # persona map (hidden attribute preference)
    persona_specs = {
        0: "helpfulness",
        1: "honesty",
    }

    # Group rows by original prompt id so we can split without leakage.
    rows_by_id = {}
    for ex in ds:
        pid = ex.get("id") or ex.get("instruction_id") or ex.get("source_id")
        if pid is None:
            # Fallback: derive from instruction text
            pid = f"hash_{abs(hash(ex.get('instruction',''))) % (10**12)}"
        rows_by_id[pid] = ex

    # Split by prompt id
    prompt_ids = list(rows_by_id.keys())
    train_ids, test_ids = split_by_prompt_id(prompt_ids, test_size, seed)

    def make_pairs_for_row(ex, pid: str) -> List[Dict[str, Any]]:
        prompt = ex.get("instruction", "")
        comps = ex.get("completions", None)
        if not prompt or not isinstance(comps, list) or len(comps) < 2:
            return []

        out = []
        for user_id, attr in persona_specs.items():
            picked = choose_pair_by_attr(comps, attr=attr, min_gap=min_gap)
            if picked is None:
                continue
            chosen, rejected, cs, rs = picked
            out.append({
                "prompt": prompt,
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                "label": 1,  # chosen preferred
                "delta_ref": float(cs - rs),  # margin under persona attr
                # Extra info (ignored by your collator, but useful for analysis)
                "meta": {
                    "user_id": int(user_id),
                    "persona_name": attr,
                    "prompt_id": pid,
                    "chosen_attr_score": cs,
                    "rejected_attr_score": rs,
                    "chosen_model": chosen.get("model"),
                    "rejected_model": rejected.get("model"),
                },
            })
        return out

    train_out, test_out = [], []
    for pid, ex in rows_by_id.items():
        pairs = make_pairs_for_row(ex, pid)
        if not pairs:
            continue
        if pid in train_ids:
            train_out.extend(pairs)
        else:
            test_out.extend(pairs)

    # Optional: deterministic shuffle within split
    rnd = random.Random(seed)
    rnd.shuffle(train_out)
    rnd.shuffle(test_out)

    # Save JSONL
    train_fp = os.path.join(out_dir, "train.jsonl")
    test_fp = os.path.join(out_dir, "test.jsonl")
    with open(train_fp, "w", encoding="utf-8") as f:
        for ex in train_out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(test_fp, "w", encoding="utf-8") as f:
        for ex in test_out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[UF-P-2] Wrote {len(train_out)} train and {len(test_out)} test examples to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="UF-P-2")
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min_gap", type=float, default=0.0,
                   help="minimum rating gap (chosen - rejected) required; set >0 to filter near-ties")
    args = p.parse_args()
    build_uf_p2(args.out_dir, test_size=args.test_size, seed=args.seed, min_gap=args.min_gap)
