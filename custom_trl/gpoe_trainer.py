from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trl.trainer.dpo_trainer import DPOTrainer  # type: ignore#
#try:
#    # TRL >= 0.7.2
#    
#except Exception as e:  # pragma: no cover
#    raise ImportError(
#        "Failed to import DPOTrainer from TRL. Ensure 'trl>=0.7.2' is installed."
#    ) from e


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
    # if delta_ref is not provided in batch and use_ref_delta is True,
    # compute it using the provided ref_model


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


class GPOETrainer(DPOTrainer):
    """
    Generalized Product-of-Experts trainer over M LoRA experts on a single base model.

    Notes
    -----
    - Single-GPU implementation (works under HF Trainer/accelerate). Multi-GPU can work
      via HF's distributed strategy, but expert loops are serialized per device.
    - Expects conventional preference batches with chosen/rejected sequences, e.g. keys like:
        - 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels'
        - 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'
        - optional: 'delta_ref' (B,) if reference log-ratio is precomputed
      If your collator uses different field names, adapt `_extract_pair_fields` accordingly.
    """

    def __init__(
        self,
        *args: Any,
        gpoe_config: Optional[GPOEConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if gpoe_config is None:
            raise ValueError("GPOETrainer requires `gpoe_config`.")
        self.gpoe_config = gpoe_config

        m = self.gpoe_config.num_experts
        if m <= 0:
            raise ValueError("num_experts must be > 0")
        if not self.gpoe_config.adapter_names or len(self.gpoe_config.adapter_names) != m:
            raise ValueError("adapter_names must be a list of length num_experts")

        # Learnable Bayesian params: mu in R^M and L lower-triangular (Cholesky) with positive diag
        device = self.accelerator.device if hasattr(self, "accelerator") else self.model.device
        self.prior = _GPOEPrior(m, device)
        # attach to model so optimizer/checkpoints track it
        setattr(self.model, "_gpoe_prior", self.prior)

        # convenience cache for expert names
        self._experts: List[str] = list(self.gpoe_config.adapter_names)

        # verify that model can switch adapters (PEFT)
        if not hasattr(self.model, "set_adapter"):
            raise ValueError(
                "Base model must be a PEFT model with multiple LoRA adapters; `model.set_adapter(name)` is required."
            )

    def get_batch_samples(self, epoch_iterator, num_batches, device=None):  # type: ignore[override]
        """Handle Trainer APIs with or without the `device` argument."""
        method = super().get_batch_samples  # type: ignore[attr-defined]
        try:
            return method(epoch_iterator, num_batches, device)  # type: ignore[misc]
        except TypeError:
            return method(epoch_iterator, num_batches)

    # ----------------------------
    # helpers
    # ----------------------------
    def _build_L(self) -> torch.Tensor:
        return self.prior.build_L()

    @staticmethod
    def _shift_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # standard causal LM shift
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
        """Compute total log probability of labeled tokens in the sequence.

        If `labels` is None, uses `input_ids` as labels; otherwise, treat -100 as ignore (masked out).
        Returns: logprob per example, shape (B,)
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if labels is None:
            labels = input_ids

        shift_logits, shift_labels = self._shift_logits_and_labels(logits, labels)
        # flatten
        vocab = shift_logits.size(-1)
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
        )
        loss = loss.view(shift_labels.size())

        # mask out ignored positions
        mask = (shift_labels != -100).float()
        token_logprobs = -loss * mask
        # sum over sequence
        return token_logprobs.sum(dim=-1)

    def _extract_pair_fields(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Support a few common field name patterns
        def req(keys: List[str]) -> List[torch.Tensor]:
            for k in keys:
                if k not in inputs:
                    raise KeyError(f"Missing required batch key: {k}")
            return [inputs[k] for k in keys]

        # Preferred keys
        try:
            (c_ids, c_mask, c_labels) = req(["chosen_input_ids", "chosen_attention_mask", "chosen_labels"])
            (r_ids, r_mask, r_labels) = req(["rejected_input_ids", "rejected_attention_mask", "rejected_labels"])
        except KeyError:
            # Fallback (some collators use *_ids, *_mask, *_labels with different prefixes)
            (c_ids, c_mask, c_labels) = req(["input_ids_chosen", "attention_mask_chosen", "labels_chosen"])
            (r_ids, r_mask, r_labels) = req(["input_ids_rejected", "attention_mask_rejected", "labels_rejected"])

        # Optional label per pair: y in {0,1}, default assume chosen preferred (1)
        label = inputs.get("pair_label", None)
        if label is None:
            label = torch.ones(c_ids.size(0), device=c_ids.device, dtype=torch.long)

        delta_ref = inputs.get("delta_ref", None)  # (B,)

        return {
            "c_ids": c_ids,
            "c_mask": c_mask,
            "c_labels": c_labels,
            "r_ids": r_ids,
            "r_mask": r_mask,
            "r_labels": r_labels,
            "label": label,
            "delta_ref": delta_ref,
        }

    # ----------------------------
    # core training step override
    # ----------------------------
    def compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False):  # type: ignore[override]
        cfg = self.gpoe_config
        m = cfg.num_experts
        beta = cfg.beta
        S = cfg.mc_samples

        fields = self._extract_pair_fields(inputs)
        c_ids, c_mask, c_labels = fields["c_ids"], fields["c_mask"], fields["c_labels"]
        r_ids, r_mask, r_labels = fields["r_ids"], fields["r_mask"], fields["r_labels"]
        pair_label = fields["label"].float()
        delta_ref = fields["delta_ref"]

        B = c_ids.size(0)
        device = c_ids.device

        # Compute reference delta if needed and not provided
        if cfg.use_ref_delta:
            if delta_ref is None:
                if self.ref_model is None:
                    raise ValueError("use_ref_delta=True but no delta_ref in batch and no ref_model provided")
                with torch.no_grad():  # ref is fixed
                    logp_c_ref = self._sequence_logprob(self.ref_model, c_ids, c_mask, c_labels)
                    logp_r_ref = self._sequence_logprob(self.ref_model, r_ids, r_mask, r_labels)
                    delta_ref = logp_c_ref - logp_r_ref
            delta_ref = delta_ref.to(device)
        else:
            delta_ref = torch.zeros(B, device=device)

        # For each expert adapter, compute log-ratio d_{i,k}
        d_list: List[torch.Tensor] = []  # list of (B,)
        for name in self._experts:
            # Activate adapter and forward both chosen and rejected
            self.model.set_adapter(name)
            logp_c = self._sequence_logprob(self.model, c_ids, c_mask, c_labels)
            logp_r = self._sequence_logprob(self.model, r_ids, r_mask, r_labels)
            d_list.append(logp_c - logp_r)

        # phi_i shape: (B, M)
        phi = torch.stack(d_list, dim=-1)

        # Sample weights via reparameterization
        mu = self.prior.mu  # (M,)
        L = self._build_L()  # (M, M)
        eps = torch.randn(S, m, device=device)
        eta = mu.unsqueeze(0) + torch.matmul(eps, L.T)  # (S, M)
        w = F.softmax(eta, dim=-1)  # (S, M)

        # s_i^(s) = beta * ((w^{(s)} · phi_i) - delta_ref)
        # Compute in a vectorized way: (B, S)
        # phi: (B, M); w.T: (M, S) => (B, S)
        ws = w.transpose(0, 1)  # (M, S)
        s = beta * (torch.matmul(phi, ws) - delta_ref.unsqueeze(-1))  # (B, S)

        # Compute log mean p over samples for labels y in {0,1}
        # log p = log sigmoid(s), log (1-p) = log sigmoid(-s)
        log_p = -F.softplus(-s)  # (B, S)
        log_1mp = -F.softplus(s)
        # select per label
        log_p_t = pair_label.unsqueeze(-1) * log_p + (1.0 - pair_label.unsqueeze(-1)) * log_1mp  # (B, S)
        # log mean over S
        loglik = torch.logsumexp(log_p_t, dim=-1) - math.log(S)  # (B,)
        L_pp = -loglik.mean()

        # Off-diagonal penalty
        # omega_bar_i = mean_s [ sigma(s) (1 - sigma(s)) ]
        p = torch.sigmoid(s)
        omega_bar = (p * (1 - p)).mean(dim=-1)  # (B,)

        # G_hat (offdiag): G_rs = (1/B) sum_i omega_i * d_{i,r} * d_{i,s}
        # compute outer products per example then average with omega weights
        # phi: (B, M); scale by sqrt(omega) to form weighted covariance
        w_sqrt = torch.sqrt(torch.clamp(omega_bar, min=0.0)).unsqueeze(-1)  # (B,1)
        phi_w = phi * w_sqrt  # (B, M)
        # batch gram: sum over i of outer(phi_w[i]) / B
        G = torch.matmul(phi_w.transpose(0, 1), phi_w) / B  # (M, M)
        offdiag = G - torch.diag(torch.diag(G))
        R_off = cfg.lambda_offdiag * torch.sum(offdiag**2)

        # Hyperpriors on mu and L
        R_hyp = cfg.hyp_mu_reg * torch.sum(mu**2) + cfg.hyp_L_reg * torch.sum(L**2)

        total_loss = L_pp + R_off + R_hyp

        # logging
        self.log({
            "loss_pp": L_pp.detach().item(),
            "loss_offdiag": R_off.detach().item(),
            "loss_hyp": R_hyp.detach().item(),
            "G_offdiag_fro": torch.norm(offdiag, p="fro").detach().item(),
            "mu_norm": torch.norm(mu).detach().item(),
            "L_diag_mean": torch.mean(torch.diag(L)).detach().item(),
        })

        if return_outputs:
            return total_loss, {
                "phi": phi.detach(),
                "delta_ref": delta_ref.detach(),
            }
        return total_loss

    # ----------------------------
    # save/load extensions
    # ----------------------------
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
    def load_gpoe_state(cls, path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        state = torch.load(path, map_location=device or "cpu")
        return state
