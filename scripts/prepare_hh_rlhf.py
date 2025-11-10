from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def longest_common_prefix(a_ids, b_ids):
    limit = min(len(a_ids), len(b_ids))
    i = 0
    while i < limit and a_ids[i] == b_ids[i]:
        i += 1
    return i


def main():
    parser = argparse.ArgumentParser(description="Prepare HH-RLHF dataset for GPOE training")
    parser.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/hh_rlhf_gpoe_train.jsonl")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples in split")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(args.dataset, split=args.split)
    total = len(ds)
    limit = total if args.max_samples <= 0 else min(args.max_samples, total)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for i in range(limit):
            ex = ds[i]
            chosen = ex["chosen"]
            rejected = ex["rejected"]

            chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)
            rejected_ids = tokenizer.encode(rejected, add_special_tokens=False)
            prefix_len = longest_common_prefix(chosen_ids, rejected_ids)

            prompt_ids = chosen_ids[:prefix_len]
            resp_pos_ids = chosen_ids[prefix_len:]
            resp_neg_ids = rejected_ids[prefix_len:]

            prompt = tokenizer.decode(prompt_ids)
            resp_pos = tokenizer.decode(resp_pos_ids)
            resp_neg = tokenizer.decode(resp_neg_ids)

            obj = {
                "prompt": prompt,
                "chosen": resp_pos,
                "rejected": resp_neg,
                "label": 1,
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {limit} examples to {out_path}")


if __name__ == "__main__":
    main()

