import yaml
from datasets import load_dataset


def load_cfg(path):
    return yaml.safe_load(open(path))


def format_example(ex):
    prompt = ex["instruction"] + (f"\n\n{ex['input']}" if ex["input"] else "")
    return {"text": f"### Instruction:\n{prompt}\n\n### Response:\n{ex['output']}"}


def splits(cfg):
    """Fixed train/eval split: the same seed gives the same examples every run."""
    ds = load_dataset(cfg["dataset"], split="train").shuffle(seed=cfg["seed"])
    n_eval, n_train = cfg["max_eval_samples"], cfg["max_train_samples"]
    ev = ds.select(range(n_eval)).map(format_example)
    tr = ds.select(range(n_eval, n_eval + n_train)).map(format_example)
    return tr, ev
