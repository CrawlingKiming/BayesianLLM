from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class PreferencePairCollator:
    tokenizer: any
    max_length: int = 1024

    def _build_pair(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        response_ids = tok.encode(response, add_special_tokens=False)
        if getattr(tok, "eos_token_id", None) is not None:
            response_ids = response_ids + [tok.eos_token_id]

        total_len = len(prompt_ids) + len(response_ids)
        if total_len > self.max_length:
            overflow = total_len - self.max_length
            prompt_ids = prompt_ids[max(0, overflow) :]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    @staticmethod
    def _pad_stack(items: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(t.size(0) for t in items)
        padded = []
        for t in items:
            if t.size(0) < max_len:
                pad = torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype)
                padded.append(torch.cat([t, pad], dim=0))
            else:
                padded.append(t)
        return torch.stack(padded, dim=0)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer
        if tok.pad_token_id is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        chosen_pairs = []
        rejected_pairs = []
        labels = []
        delta_refs = []
        has_delta_ref = True

        for ex in batch:
            prompt = ex.get("prompt")
            chosen = ex.get("chosen")
            rejected = ex.get("rejected")
            if prompt is None or chosen is None or rejected is None:
                raise ValueError("Examples must include 'prompt', 'chosen', 'rejected'")

            chosen_pairs.append(self._build_pair(prompt, chosen))
            rejected_pairs.append(self._build_pair(prompt, rejected))
            labels.append(int(ex.get("label", 1)))

            if "delta_ref" in ex and ex["delta_ref"] is not None:
                delta_refs.append(float(ex["delta_ref"]))
            else:
                has_delta_ref = False

        batch_out = {
            "chosen_input_ids": self._pad_stack([x["input_ids"] for x in chosen_pairs], tok.pad_token_id),
            "chosen_attention_mask": self._pad_stack([x["attention_mask"] for x in chosen_pairs], 0),
            "chosen_labels": self._pad_stack([x["labels"] for x in chosen_pairs], -100),
            "rejected_input_ids": self._pad_stack([x["input_ids"] for x in rejected_pairs], tok.pad_token_id),
            "rejected_attention_mask": self._pad_stack([x["attention_mask"] for x in rejected_pairs], 0),
            "rejected_labels": self._pad_stack([x["labels"] for x in rejected_pairs], -100),
            "pair_label": torch.tensor(labels, dtype=torch.long),
        }

        if has_delta_ref and len(delta_refs) == len(batch):
            batch_out["delta_ref"] = torch.tensor(delta_refs, dtype=torch.float)

        return batch_out

