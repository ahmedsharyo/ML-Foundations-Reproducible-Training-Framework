from pathlib import Path
import json
import matplotlib.pyplot as plt

def plot_history(path, output):
    data=json.loads(Path(path).read_text(encoding="utf8")); rows=data["history"]; epochs=[r["epoch"] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].plot(epochs,[r["train_loss"] for r in rows],label="train"); axes[0].plot(epochs,[r["val_loss"] for r in rows],label="validation"); axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(epochs,[r["train_accuracy"] for r in rows],label="train"); axes[1].plot(epochs,[r["val_accuracy"] for r in rows],label="validation"); axes[1].set_title("Accuracy"); axes[1].legend()
    fig.tight_layout(); Path(output).parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,dpi=150); plt.close(fig)
