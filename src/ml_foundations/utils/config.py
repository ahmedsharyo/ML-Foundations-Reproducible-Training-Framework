from pathlib import Path
import yaml

def load_config(path):
    with Path(path).open(encoding="utf8") as f: config=yaml.safe_load(f)
    if not isinstance(config,dict): raise ValueError("configuration root must be a mapping")
    return config
