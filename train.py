"""QLoRA fine-tune. Needs a GPU (a free Colab T4 is enough for the default config)."""
import argparse

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from common import load_cfg, splits

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
cfg = load_cfg(ap.parse_args().config)

tok = AutoTokenizer.from_pretrained(cfg["model"])
quant = None
if cfg["quantize_4bit"]:
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
model = AutoModelForCausalLM.from_pretrained(cfg["model"], quantization_config=quant, device_map="auto")

train_ds, _ = splits(cfg)

args = SFTConfig(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["epochs"],
    per_device_train_batch_size=cfg["batch_size"],
    gradient_accumulation_steps=cfg["grad_accum"],
    learning_rate=float(cfg["lr"]),  # YAML may hand back "2e-4" as a string
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="mlflow",
    seed=cfg["seed"],
)
lora = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"],
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, peft_config=lora, processing_class=tok)
trainer.train()
trainer.model.save_pretrained(cfg["output_dir"])
print("adapter saved to", cfg["output_dir"])
