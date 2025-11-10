from __future__ import annotations

import argparse
from typing import List, Optional

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from datasets import load_dataset

from peft import LoraConfig, get_peft_model

from custom_trl.gpoe_trainer import GPOETrainer, GPOEConfig
from custom_trl.collators import PreferencePairCollator


def attach_lora_experts(
    model,
    adapter_names: List[str],
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

        model = get_peft_model(model, lora_cfg, adapter_name=adapter_names[0])
        for name in adapter_names[1:]:
            model.add_adapter(name, lora_cfg)

        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name

        return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPOE with LoRA experts")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ref_model_name", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_experts", type=int, required=True)
    parser.add_argument("--adapter_names", type=str, nargs="+", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--mc_samples", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda_offdiag", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    assert len(args.adapter_names) == args.num_experts, "adapter_names must match num_experts"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_name)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    model = attach_lora_experts(base_model, args.adapter_names)

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_path},
        keep_in_memory=False,
    )
    train_ds = dataset["train"]
    required_fields = {"prompt", "chosen", "rejected"}
    missing = required_fields - set(train_ds.column_names)
    if missing:
        raise ValueError(f"Dataset missing required fields: {sorted(missing)}")

    collator = PreferencePairCollator(tokenizer=tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to=["none"],
        seed=args.seed,
    )

    gpoe_cfg = GPOEConfig(
        num_experts=args.num_experts,
        adapter_names=args.adapter_names,
        beta=args.beta,
        mc_samples=args.mc_samples,
        lambda_offdiag=args.lambda_offdiag,
        use_ref_delta=True,
        hyp_mu_reg=0.0,
        hyp_L_reg=0.0,
    )

    trainer = GPOETrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tokenizer,
        gpoe_config=gpoe_cfg,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
