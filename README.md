# lora-finetune-playbook

LoRA/QLoRA fine-tuning of a small open LLM with tracked experiments and a before/after evaluation.

```bash
pip install -r requirements.txt
python train.py --config configs/qwen-1.5b-lora.yaml
python evaluate.py --config configs/qwen-1.5b-lora.yaml --adapter outputs/run1
```

Config:

```yaml
model: Qwen/Qwen2.5-1.5B-Instruct
dataset: yahma/alpaca-cleaned
max_train_samples: 2000
lora: {r: 16, alpha: 32, dropout: 0.05}
lr: 2.0e-4
epochs: 1
quantize_4bit: true
```

## Results

Held-out perplexity on 200 examples the model never trained on (lower is better).

| Model | Perplexity |
|---|---|
| Base | - |
| Fine-tuned | - |

Fill this in from `evaluate.py`.

## Notes

- Built with Hugging Face PEFT and TRL
- Training loss is logged to MLflow (`mlflow ui` to browse runs)
- 4-bit loading is meant to fit a free Colab T4
- The train/eval split is fixed by the seed in the config
- Perplexity only shows the model fits the training format better; add a task metric or an LLM judge for a real quality claim

MIT
