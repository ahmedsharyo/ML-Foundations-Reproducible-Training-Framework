# ML Foundations: Reproducible Training Framework

A compact portfolio project implementing linear and logistic regression with NumPy and comparing an MLP, compact CNN, and small Vision Transformer on Fashion-MNIST. The code favors inspectable algorithms and explicit experiment state over a large training framework.

## What you will learn

- How regression and binary classification objectives produce analytical gradients.
- How PyTorch autograd, modules, optimizers, and evaluation mode fit into a shared training loop.
- How to keep data partitions, random initialization, optimizer comparisons, checkpoints, and metrics reproducible.
- How convolution and patch-based self-attention represent images differently.

## Models and mathematics

**Linear regression (implemented manually).** For a feature matrix `X`, weights `w`, and bias `b`, predictions are:

```text
predictions = X @ w + b
```

The objective is mean squared error with optional L2 regularization:

```text
loss = sum((prediction - target)^2) / (2 * n) + lambda * sum(w^2) / 2
gradient_w = X.T @ (prediction - target) / n + lambda * w
gradient_b = mean(prediction - target)
```

Full-batch and shuffled mini-batch updates are supported.

**Logistic regression (implemented manually).** For one example with feature vector `x`, the predicted probability of class 1 is `sigmoid(x @ w + b)`. The sign-split sigmoid implementation avoids overflow. Binary cross-entropy with L2 regularization is:

```text
loss = -mean(y * log(p) + (1 - y) * log(1 - p)) + lambda * sum(w^2) / 2
gradient_w = X.T @ (p - y) / n + lambda * w
gradient_b = mean(p - y)
```

The model returns probabilities and thresholded class labels. Both NumPy estimators are checked against central finite differences, and their benchmark scores are compared with scikit-learn.

**MLP.** Flatten the input image, apply configurable Linear→ReLU blocks (optional dropout), and emit one logit per class. Image size, input channels, and class count are configurable; defaults match Fashion-MNIST's 28×28 grayscale images and ten classes. PyTorch computes gradients by autograd; cross-entropy consumes logits.

**CNN.** Repeated 3×3 convolution, ReLU, and 2×2 max-pooling blocks feed a compact dense classifier.

**TinyViT.** Non-overlapping image patches are embedded and combined with learned position and class tokens. Each transformer block applies pre-normalized multi-head self-attention and a feed-forward residual block. For a batch of `B` images and `N` patches, token tensors have shape `[B, N+1, D]`; attention weights (if requested) have shape `[B, heads, N+1, N+1]`. Scaled dot-product attention works in three steps:

```text
attention(Q, K, V) = softmax((Q @ K.T) / sqrt(key_dimension)) @ V
```

The implementation uses PyTorch `nn.MultiheadAttention`.

Gradient descent updates parameters by subtracting the learning rate times the loss gradient:

```text
new_parameters = current_parameters - learning_rate * gradient(loss)
```

The NumPy models calculate these derivatives directly; PyTorch models use autograd. Optimizers are SGD, SGD with momentum, and Adam.

## Setup

Python 3.11 or newer is required. On Apple Silicon, install a current PyTorch/torchvision wheel compatible with your Python version; PyTorch provides MPS support in its macOS packages. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The dependency minimums are in `pyproject.toml`; record the exact resolved environment with `python -m pip freeze` when publishing experiment results. Fashion-MNIST is downloaded by torchvision on first run into `data/` (free public dataset). Tests use synthetic data and do not download it.

## Commands

```bash
python scripts/train.py --config configs/linear.yaml
python scripts/train.py --config configs/logistic.yaml
python scripts/train.py --config configs/mlp.yaml
python scripts/train.py --config configs/cnn.yaml
python scripts/train.py --config configs/attention.yaml
python scripts/compare_optimizers.py --config configs/optimizer_comparison.yaml
python scripts/evaluate.py --checkpoint checkpoints/mlp_best.pt
pytest -q
```

For a short CPU-only end-to-end run, use `python scripts/train.py --config configs/smoke.yaml`. PyTorch checkpoints are saved under `checkpoints/`; JSON histories and environment metadata go under `results/`. A resumed run uses `python scripts/train.py --config configs/mlp.yaml --resume checkpoints/mlp_best.pt`. Checkpoints retain model and optimizer state, epoch, configuration, history, best validation loss, and PyTorch RNG state.

## Reproducibility and hardware

The configured seed covers Python, NumPy, and PyTorch. Fashion-MNIST train/validation partitioning and DataLoader shuffling use seeded generators. Each comparison arm starts from the same seed, architecture, split, and epoch budget; only the optimizer differs. The best validation-loss checkpoint is selected and evaluated once on the held-out test set. Output JSON contains per-epoch loss, accuracy, learning rate, elapsed training time, selected test metrics, device, and Python/PyTorch versions.

`device: auto` selects MPS when available and otherwise CPU; `device: cpu` provides a practical debugging/reproduction path. MPS operators may differ numerically from CPU, and exact cross-device or cross-version determinism is not promised. Unsupported MPS operations can be worked around by selecting CPU. Use `num_workers=0` on macOS unless a different setting has been verified locally.

## Experiments and interpretation

Run the optimizer comparison to generate actual metrics before drawing conclusions. The experiment uses one fixed MLP and fixed data split; inspect each JSON history for loss convergence, validation accuracy, training time, epoch count, and train/validation accuracy gap. Plot the per-run histories with `python scripts/plot_results.py --input results/optimizer_sgd.json results/optimizer_momentum.json results/optimizer_adam.json`. SGD is simple and can require careful learning-rate tuning; momentum smooths updates and can accelerate progress; Adam adapts per-parameter step sizes and often converges quickly, with additional optimizer state and potentially different generalization. These are qualitative expectations, not fabricated benchmark findings. This repository does not commit benchmark numbers; generated results are local artifacts and should only be published after running the commands on the target hardware.

## Tests and limitations

`pytest -q` checks NumPy gradients and sigmoid stability, model forward/backward shapes, deterministic splitting, device selection, checkpoint round trips, and a synthetic training run. Fashion-MNIST downloads are not part of the tests. The included framework intentionally omits schedulers, mixed precision, distributed training, and broad dataset abstractions. Fashion-MNIST performance depends on training budget and hardware. Torchvision/PyTorch binary compatibility must match the local Python and macOS installation.

Possible next steps include adding a plotting command for JSON histories, richer classification metrics, experiment manifests with full package freeze output, and additional dataset/model baselines.
