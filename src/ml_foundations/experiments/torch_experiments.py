from __future__ import annotations

import json
import platform
from pathlib import Path

import torch
from torch import nn

from ..data.fashion_mnist import make_loaders
from ..models.constants import DEFAULT_IMAGE_SIZE
from ..models.torch_models import MLP, FashionCNN, TinyViT
from ..training.engine import fit, evaluate, load_checkpoint
from ..utils.reproducibility import get_device, set_seed


def build_model(config):
    m=config.get("model",{}); kind=m.get("type","mlp"); args={k:v for k,v in m.items() if k!="type"}
    return {"mlp":MLP,"cnn":FashionCNN,"attention":TinyViT}[kind](**args)

def run(config, resume=None):
    seed=config.get("seed",42); set_seed(seed); device=get_device(config.get("device","auto")); t=config.get("training",{})
    train,val,test=make_loaders(config.get("data_dir","data"),
                                t.get("batch_size",128),seed,
                                limits=t.get("limit"),image_size=config.get("model",{}).get("image_size",DEFAULT_IMAGE_SIZE))
    model=build_model(config).to(device); opt=t.get("optimizer",{}); name=opt.get("name","adam").lower();
    lr=opt.get("learning_rate",.001); wd=opt.get("weight_decay",0.)
    if name=="sgd": optimizer=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
    elif name in ("momentum","sgd_momentum"): optimizer=torch.optim.SGD(model.parameters(),
                                                                        lr=lr,momentum=opt.get("momentum",.9),weight_decay=wd)
    elif name=="adam": optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    else: raise ValueError(f"Unsupported optimizer: {name}")
    out=Path(config.get("output_dir","results")); ckpt=Path(config.get("checkpoint_dir","checkpoints"))
    out.mkdir(parents=True,exist_ok=True); ckpt.mkdir(parents=True,exist_ok=True)
    stem=config.get("name",config["model"]["type"]); cp=ckpt/f"{stem}_best.pt"
    result=fit(model,train,val,optimizer,nn.CrossEntropyLoss(),device,t.get("epochs",5),cp,config,resume)
    load_checkpoint(cp,model,map_location=device);
    result["test"]=evaluate(model,test,nn.CrossEntropyLoss(),device);
    result.update({"device":str(device),"python":platform.python_version(),"torch":torch.__version__})
    result["optimizer"]=name
    (out/f"{stem}.json").write_text(json.dumps(result,indent=2),encoding="utf8"); print(json.dumps(result,indent=2)); return result
