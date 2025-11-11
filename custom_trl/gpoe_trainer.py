from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trl.trainer.base_trainer import BaseTrainer


@dataclass
class GPOEConfig:
    num_experts: int
    adapter_names: List[str]
    beta: float = 1.0
    mc_samples: int = 8
    lambda_offdiag: float = 0.0
    use_ref_delta: bool = True
    hyp_mu_reg: float = 0.0
    hyp_L_reg: float = 0.0


class _GPOEPrior(nn.Module):
    def __init__(self, num_experts: int, device: torch.device):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(num_experts, device=device))
        self.L_raw = nn.Parameter(torch.zeros(num_experts, num_experts, device=device))

    def build_L(self) -> torch.Tensor:
        m = self.mu.shape[0]
        tril = torch.tril(self.L_raw)
        diag = F.softplus(torch.diag(tril))
        return tril - torch.diag(torch.diag(tril)) + torch.diag(diag)


class GPOETrainer(BaseTrainer):
    """Trainer that implements the GPoE loss on top of TRL's BaseTrainer."""

    def __init__(
        self,
        *args: Any,
        gpoe_config: Optional[GPOEConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.ref_model = kwargs.pop("ref_model", None)
        super().__init__(*args, **kwargs)

        if gpoe_config is None:
            raise ValueError("GPOETrainer requires `gpoe_config`.")
        self.gpoe_config = gpoe_config

        m = self.gpoe_config.num_experts
        if m <= 0:
            raise ValueError("num_experts must be > 0")
        if not self.gpoe_config.adapter_names or len(self.gpoe_config.adapter_names) != m:
            raise ValueError("adapter_names must be a list of length num_experts")

        device = self.accelerator.device if hasattr(self, "accelerator") else self.model.device
        self.prior = _GPOEPrior(m, device)
        setattr(self.model, "_gpoe_prior", self.prior)

        if self.ref_model is not None:
            self.ref_model.to(device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self._experts: List[str] = list(self.gpoe_config.adapter_names)
        if not hasattr(self.model, "set_adapter"):
            raise ValueError(
                "Base model must be a PEFT model with multiple LoRA adapters; `model.set_adapter(name)` is required."
            )

    # helper utilities --------------------------------------------------
    def _build_L(self) -> torch.Tensor:
        return self.prior.build_L()

    @staticmethod
    def _shift_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return shift_logits, shift_labels

    def _sequence_logprob(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        labels = input_ids if labels is None else labels
        shift_logits, shift_labels = self._shift_logits_and_labels(logits, labels)
        vocab = shift_logits.size(-1)
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
        )
        loss = loss.view(shift_labels.size())
        mask = (shift_labels != -100).float()
        token_logprobs = -loss * mask
        return token_logprobs.sum(dim=-1)

    def _extract_pair_fields(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        def req(keys: List[str]) -> List[torch.Tensor]:
            missing = [k for k in keys if k not in inputs]
            if missing:
                raise KeyError(f"Missing required batch key(s): {missing}")
            return [inputs[k] for k in keys]

        try:
            (c_ids, c_mask, c_labels) = req(["chosen_input_ids", "chosen_attention_mask", "chosen_labels"])
            (r_ids, r_mask, r_labels) = req(["rejected_input_ids", "rejected_attention_mask", "rejected_labels"])
        except KeyError:
            (c_ids, c_mask, c_labels) = req(["input_ids_chosen", "attention_mask_chosen", "labels_chosen"])
            (r_ids, r_mask, r_labels) = req(["input_ids_rejected", "attention_mask_rejected", "labels_rejected"])

        label = inputs.get("pair_label")
        if label is None:
            label = torch.ones(c_ids.size(0), device=c_ids.device, dtype=torch.long)

        delta_ref = inputs.get("delta_ref")
        return {
            "c_ids": c_ids,
            "c_mask": c_mask,
            "c_labels": c_labels,
            "r_ids": r_ids,
            "r_mask": r_mask,
            "r_labels": r_labels,
            "label": label.float(),
            "delta_ref": delta_ref,
        }

    # core training step ------------------------------------------------
    def compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False):  # type: ignore[override]
        cfg = self.gpoe_config
        fields = self._extract_pair_fields(inputs)
        c_ids, c_mask, c_labels = fields["c_ids"], fields["c_mask"], fields["c_labels"]
        r_ids, r_mask, r_labels = fields["r_ids"], fields["r_mask"], fields["r_labels"]
        pair_label = fields["label"]
        delta_ref = fields["delta_ref"]

        B = c_ids.size(0)
        device = c_ids.device

        if cfg.use_ref_delta:
            if delta_ref is None:
                if self.ref_model is None:
                    raise ValueError("use_ref_delta=True but no delta_ref in batch and no ref_model provided")
                with torch.no_grad():
                    logp_c_ref = self._sequence_logprob(self.ref_model, c_ids, c_mask, c_labels)
                    logp_r_ref = self._sequence_logprob(self.ref_model, r_ids, r_mask, r_labels)
                    delta_ref = logp_c_ref - logp_r_ref
            delta_ref = delta_ref.to(device)
        else:
            delta_ref = torch.zeros(B, device=device)

        d_list: List[torch.Tensor] = []
        for name in self._experts:
            self.model.set_adapter(name)
            logp_c = self._sequence_logprob(self.model, c_ids, c_mask, c_labels)
            logp_r = self._sequence_logprob(self.model, r_ids, r_mask, r_labels)
            d_list.append(logp_c - logp_r)
        phi = torch.stack(d_list, dim=-1)

        m = cfg.num_experts
        S = cfg.mc_samples
        beta = cfg.beta

        mu = self.prior.mu
        L = self._build_L()
        eps = torch.randn(S, m, device=device)
        eta = mu.unsqueeze(0) + torch.matmul(eps, L.T)
        w = F.softmax(eta, dim=-1)
        ws = w.transpose(0, 1)
        s = beta * (torch.matmul(phi, ws) - delta_ref.unsqueeze(-1))

        log_p = -F.softplus(-s)
        log_1mp = -F.softplus(s)
        log_p_t = pair_label.unsqueeze(-1) * log_p + (1.0 - pair_label.unsqueeze(-1)) * log_1mp
        loglik = torch.logsumexp(log_p_t, dim=-1) - math.log(S)
        L_pp = -loglik.mean()

        p = torch.sigmoid(s)
        omega_bar = (p * (1 - p)).mean(dim=-1)
        w_sqrt = torch.sqrt(torch.clamp(omega_bar, min=0.0)).unsqueeze(-1)
        phi_w = phi * w_sqrt
        G = torch.matmul(phi_w.transpose(0, 1), phi_w) / B
        offdiag = G - torch.diag(torch.diag(G))
        R_off = cfg.lambda_offdiag * torch.sum(offdiag**2)

        R_hyp = cfg.hyp_mu_reg * torch.sum(mu**2) + cfg.hyp_L_reg * torch.sum(L**2)
        total_loss = L_pp + R_off + R_hyp

        self.log({
            "loss_pp": float(L_pp.detach()),
            "loss_offdiag": float(R_off.detach()),
            "loss_hyp": float(R_hyp.detach()),
            "G_offdiag_fro": float(torch.norm(offdiag, p="fro").detach()),
            "mu_norm": float(torch.norm(mu).detach()),
            "L_diag_mean": float(torch.mean(torch.diag(L)).detach()),
        })

        if return_outputs:
            return total_loss, {"phi": phi.detach(), "delta_ref": delta_ref.detach()}
        return total_loss

    # checkpoint helpers -----------------------------------------------
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):  # type: ignore[override]
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        target_dir = output_dir or self.args.output_dir
        state = {
            "mu": self.prior.mu.detach().cpu(),
            "L_raw": self.prior.L_raw.detach().cpu(),
            "adapter_names": self._experts,
            "gpoe_config": self.gpoe_config.__dict__,
        }
        torch.save(state, f"{target_dir}/gpoe_state.pt")

    @classmethod
    @classmethod
    def load_gpoe_state(cls, path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        state = torch.load(path, map_location=device or "cpu")
        return state
