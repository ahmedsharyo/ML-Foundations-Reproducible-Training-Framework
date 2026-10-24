from __future__ import annotations
import json, time
from pathlib import Path
import torch
from torch import nn

def _run(model,loader,criterion,device,optimizer=None):
    train=optimizer is not None; model.train(train); total=correct=count=0; loss_sum=0.
    context=torch.enable_grad() if train else torch.no_grad()
    with context:
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            if train: optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,y)
            if train: loss.backward(); optimizer.step()
            n=y.size(0); count+=n; loss_sum+=loss.item()*n; correct+=(logits.argmax(1)==y).sum().item()
    if not count: raise ValueError("loader produced no batches")
    return {"loss":loss_sum/count,"accuracy":correct/count}

def train_one_epoch(model,loader,optimizer,criterion,device): return _run(model,loader,criterion,device,optimizer)
def evaluate(model,loader,criterion,device): return _run(model,loader,criterion,device)
def predict(model,loader,device):
    model.eval(); outputs=[]
    with torch.no_grad():
        for x,*_ in loader: outputs.append(model(x.to(device)).argmax(1).cpu())
    return torch.cat(outputs) if outputs else torch.empty(0,dtype=torch.long)

def save_checkpoint(path,model,optimizer,epoch,config,history=None,best_metric=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict() if optimizer else None,"epoch":epoch,"config":config,"history":history or [],"best_metric":best_metric,"torch_rng":torch.get_rng_state()},path)

def load_checkpoint(path,model,optimizer=None,map_location="cpu"):
    state=torch.load(path,map_location=map_location,weights_only=False); model.load_state_dict(state["model_state"])
    if optimizer is not None and state.get("optimizer_state") is not None: optimizer.load_state_dict(state["optimizer_state"])
    return state

def fit(model,train_loader,val_loader,optimizer,criterion,device,epochs,checkpoint_path,config,resume=None):
    start=0; history=[]; best=float("inf")
    if resume:
        state=load_checkpoint(resume,model,optimizer,device); start=state["epoch"]; history=state.get("history",[]); best=state.get("best_metric",best)
    model.to(device); started=time.perf_counter()
    for epoch in range(start,epochs):
        tr=train_one_epoch(model,train_loader,optimizer,criterion,device); va=evaluate(model,val_loader,criterion,device)
        row={"epoch":epoch+1,"train_loss":tr["loss"],"train_accuracy":tr["accuracy"],"val_loss":va["loss"],"val_accuracy":va["accuracy"],"learning_rate":optimizer.param_groups[0]["lr"]}; history.append(row)
        if va["loss"]<best:
            best=va["loss"]; save_checkpoint(checkpoint_path,model,optimizer,epoch+1,config,history,best)
    return {"history":history,"best_val_loss":best,"training_seconds":time.perf_counter()-started}
