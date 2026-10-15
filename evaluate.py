"""Held-out perplexity of the base model vs base + LoRA adapter (lower is better)."""
import argparse
import math

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_cfg, splits

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--adapter", required=True, help="folder written by train.py")
a = ap.parse_args()
cfg = load_cfg(a.config)

tok = AutoTokenizer.from_pretrained(cfg["model"])
_, eval_ds = splits(cfg)
base = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float16, device_map="auto")


@torch.no_grad()
def perplexity(model):
    model.eval()
    nll, n = 0.0, 0
    for ex in eval_ds:
        ids = tok(ex["text"], return_tensors="pt", truncation=True, max_length=cfg["max_length"]).to(model.device)
        loss = model(**ids, labels=ids["input_ids"]).loss
        k = ids["input_ids"].numel() - 1  # tokens actually predicted
        nll += loss.item() * k
        n += k
    return math.exp(nll / n)


print(f"base perplexity:       {perplexity(base):.3f}")
tuned = PeftModel.from_pretrained(base, a.adapter)
print(f"fine-tuned perplexity: {perplexity(tuned):.3f}")
