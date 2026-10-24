#!/usr/bin/env python3
import argparse

import torch
from torch import nn

from ml_foundations.data.fashion_mnist import make_loaders
from ml_foundations.experiments.torch_experiments import build_model
from ml_foundations.training.engine import load_checkpoint, evaluate
from ml_foundations.utils.reproducibility import get_device


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--device",default="auto"); p.add_argument("--data-dir",default="data"); args=p.parse_args()
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False); cfg=state["config"]; device=get_device(args.device); model=build_model(cfg).to(device); load_checkpoint(args.checkpoint,model,map_location=device)
    _,_,test=make_loaders(args.data_dir,cfg.get("training",{}).get("batch_size",128),cfg.get("seed",42)); print(evaluate(model,test,nn.CrossEntropyLoss(),device))
if __name__=="__main__": main()
