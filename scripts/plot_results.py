#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description="Plot training curves and optimizer comparison JSON")
    p.add_argument("--input",nargs="+",required=True,help="One or more result JSON files")
    p.add_argument("--output",default="results/comparison.png"); args=p.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    summary=[]
    for filename in args.input:
        data=json.loads(Path(filename).read_text(encoding="utf8")); rows=data.get("history",[]); label=data.get("optimizer",Path(filename).stem)
        if rows:
            epochs=[r["epoch"] for r in rows]; axes[0].plot(epochs,[r["train_loss"] for r in rows],label=f"{label} train"); axes[0].plot(epochs,[r["val_loss"] for r in rows],linestyle="--",label=f"{label} val")
        if "test" in data: summary.append((label,data["test"]["accuracy"],data.get("training_seconds",0)))
        elif "test_accuracy" in data: summary.append((label,data["test_accuracy"],data.get("training_seconds",0)))
    axes[0].set(xlabel="Epoch",ylabel="Cross-entropy",title="Loss convergence"); axes[0].legend(fontsize=8)
    if summary:
        axes[1].bar([x[0] for x in summary],[x[1] for x in summary]); axes[1].set(ylabel="Test accuracy",title="Held-out comparison"); axes[1].tick_params(axis="x",rotation=20)
    else: axes[1].text(.5,.5,"Use per-run result JSON files",ha="center",va="center"); axes[1].set_axis_off()
    fig.tight_layout(); Path(args.output).parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=150); plt.close(fig)
if __name__=="__main__": main()
