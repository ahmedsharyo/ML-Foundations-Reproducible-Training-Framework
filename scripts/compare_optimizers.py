#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

from ml_foundations.experiments.torch_experiments import run
from ml_foundations.utils.config import load_config


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); args=p.parse_args(); base=load_config(args.config); results=[]
    for opt in ("sgd","momentum","adam"):
        cfg=copy.deepcopy(base); cfg["name"]=f"optimizer_{opt}"; cfg["training"]["optimizer"]["name"]=opt
        results.append(run(cfg))
    output=Path(base.get("output_dir","results")); output.mkdir(parents=True,exist_ok=True); (output/"optimizer_comparison.json").write_text(json.dumps(results,indent=2),encoding="utf8")
if __name__=="__main__": main()
