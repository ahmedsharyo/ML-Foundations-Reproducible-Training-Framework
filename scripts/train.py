#!/usr/bin/env python3
import argparse
from ml_foundations.utils.config import load_config
from ml_foundations.experiments.numpy_experiments import run as run_numpy
from ml_foundations.experiments.torch_experiments import run as run_torch

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--resume"); args=p.parse_args(); cfg=load_config(args.config)
    kind=cfg.get("model",{}).get("type")
    if kind in ("linear","logistic"): run_numpy(kind,cfg)
    else: run_torch(cfg,args.resume)
if __name__=="__main__": main()
