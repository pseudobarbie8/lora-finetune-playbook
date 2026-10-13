# lora-finetune-playbook

LoRA/QLoRA fine-tuning of a small open LLM with tracked experiments and before/after evals.

```bash
pip install -r requirements.txt
python train.py --config configs/qwen-1.5b-lora.yaml
python evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/run1
```

Config example:

```yaml
model: Qwen/Qwen2.5-1.5B-Instruct
dataset: yahma/alpaca-cleaned
lora: {r: 16, alpha: 32, dropout: 0.05}
lr: 2e-4
epochs: 1
quantize_4bit: true
```

## Results

| Model | Metric | Score |
|---|---|---|
| Base | - | - |
| Fine-tuned | - | - |

Fill this in after `evaluate.py`.

## Notes

- Built with Hugging Face PEFT and TRL
- Runs are logged to MLflow (`mlflow ui` to browse)
- 4-bit fits on a free Colab T4
- Same seed and config should reproduce a run

MIT
