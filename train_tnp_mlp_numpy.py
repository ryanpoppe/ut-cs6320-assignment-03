"""CS 6320 Assignment 3 - Part A: NumPy twin of the training loop.

This is a dependency-free companion to ``train_tnp_mlp.py`` (PyTorch). It builds
the *same* small MLP (one hidden ReLU layer -> single regression output), trains
it with the *same* full-batch loop, the same target standardization, the same
early stopping, and writes the same artifact format.

The only difference is that here the backward pass is written out by hand (the
chain rule, step by step) instead of being produced by ``loss.backward()``.
PyTorch's autograd automates exactly this calculation; coding it manually is the
clearest possible way to show I understand what backpropagation is doing.

I use this script to generate the run evidence (metrics, training history, the
training-curve plot, and the log) on a machine without a PyTorch install. The
CHPC submission runs the PyTorch version, which produces equivalent behavior.

Run:
    python train_tnp_mlp_numpy.py
    python train_tnp_mlp_numpy.py --optimizer sgd --learning-rate 0.1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np

from tnp_common import (
    DEFAULT_PREP_DIR,
    RANDOM_STATE,
    build_preprocessor,
    default_run_dir,
    evaluate_predictions,
    load_splits,
    split_xy,
    write_json,
    write_metrics_summary,
    write_run_metadata,
)


# --------------------------------------------------------------------------- #
# A one-hidden-layer MLP with explicit forward and backward passes.
# --------------------------------------------------------------------------- #
class NumpyMLP:
    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.Generator) -> None:
        # He initialization for the ReLU layer; small init for the output layer.
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * math.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, 1)) * math.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros(1)
        self._cache: dict[str, np.ndarray] = {}

    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """FORWARD PASS: inputs -> hidden ReLU -> linear output (shape (n,))."""
        z1 = x @ self.W1 + self.b1          # pre-activation
        a1 = np.maximum(z1, 0.0)            # ReLU activation
        out = a1 @ self.W2 + self.b2        # (n, 1)
        self._cache = {"x": x, "z1": z1, "a1": a1}
        return out.ravel()

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
        """BACKPROP by hand: gradient of MSE loss w.r.t. every parameter.

        This is the chain rule applied layer by layer - exactly what
        ``loss.backward()`` computes automatically in the PyTorch version.
        """
        x, z1, a1 = self._cache["x"], self._cache["z1"], self._cache["a1"]
        n = y_true.shape[0]

        # d(MSE)/d(pred); MSE = mean((pred - t)^2)
        d_out = (2.0 / n) * (y_pred - y_true).reshape(-1, 1)   # (n, 1)

        # Output linear layer: out = a1 @ W2 + b2
        dW2 = a1.T @ d_out                                     # (hidden, 1)
        db2 = d_out.sum(axis=0)                                # (1,)

        # Back through the hidden layer
        d_a1 = d_out @ self.W2.T                               # (n, hidden)
        d_z1 = d_a1 * (z1 > 0.0)                               # ReLU derivative
        dW1 = x.T @ d_z1                                       # (input, hidden)
        db1 = d_z1.sum(axis=0)                                 # (hidden,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# --------------------------------------------------------------------------- #
# Optimizers (hand-written, mirroring torch.optim)
# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for k in params:
            params[k] -= self.lr * grads[k]  # w <- w - lr * dL/dw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", default=str(DEFAULT_PREP_DIR))
    parser.add_argument("--output-dir", help="Defaults to <prep-dir>/runs/small_mlp_numpy/.")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def mse_std(y_true_std: np.ndarray, y_pred_std: np.ndarray) -> float:
    return float(np.mean((y_pred_std - y_true_std) ** 2))


def grad_global_norm(grads: dict[str, np.ndarray]) -> float:
    return math.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))


def main() -> None:
    args = parse_args()
    prep_dir = Path(args.prep_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else default_run_dir(prep_dir, "small_mlp_numpy")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- Data + preprocessing (shared with baselines + torch version) ----- #
    split_pairs = split_xy(load_splits(prep_dir))
    x_train_df, y_train = split_pairs["train"]
    x_val_df, y_val = split_pairs["validation"]

    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(x_train_df).astype(np.float64)
    x_val = preprocessor.transform(x_val_df).astype(np.float64)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    y_train_sec = y_train.to_numpy(dtype=np.float64)
    y_val_sec = y_val.to_numpy(dtype=np.float64)

    # Standardize the target for stable MSE training; keep mean/std to invert.
    y_mean = float(y_train_sec.mean())
    y_std = float(y_train_sec.std())
    y_train_std = (y_train_sec - y_mean) / y_std
    y_val_std = (y_val_sec - y_mean) / y_std

    # ---- Model + optimizer ------------------------------------------------ #
    model = NumpyMLP(x_train.shape[1], args.hidden_dim, rng)
    params = model.params()
    optimizer = Adam(params, args.learning_rate) if args.optimizer == "adam" else SGD(args.learning_rate)
    n_params = sum(p.size for p in params.values())

    print(f"Model: NumpyMLP(input_dim={x_train.shape[1]}, hidden_dim={args.hidden_dim}) "
          f"-> {n_params:,} trainable parameters")
    print(f"Optimizer: {args.optimizer} (lr={args.learning_rate}) | loss: MSE on standardized target")
    print(f"Training full-batch for up to {args.epochs} epochs (patience={args.patience})\n")

    history: list[dict] = []
    best_params = None
    best_val_loss = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # (1) FORWARD
        pred_train_std = model.forward(x_train)
        # (2) LOSS
        train_loss = mse_std(y_train_std, pred_train_std)
        # (3+4) BACKPROP (hand-coded chain rule fills the gradients)
        grads = model.backward(y_train_std, pred_train_std)
        grad_norm = grad_global_norm(grads)
        # (5) PARAMETER UPDATE
        optimizer.step(params, grads)

        # VALIDATION TRACKING
        pred_val_std = model.forward(x_val)
        val_loss = mse_std(y_val_std, pred_val_std)

        train_mae = evaluate_predictions(y_train_sec, model.forward(x_train) * y_std + y_mean)["mae_seconds"]
        val_mae = evaluate_predictions(y_val_sec, pred_val_std * y_std + y_mean)["mae_seconds"]

        history.append({
            "epoch": epoch,
            "train_loss_std": train_loss,
            "val_loss_std": val_loss,
            "grad_norm": grad_norm,
            "train_mae_seconds": train_mae,
            "val_mae_seconds": val_mae,
        })

        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch {epoch:4d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                  f"grad_norm {grad_norm:7.3f} | train_MAE {train_mae:6.1f}s | val_MAE {val_mae:6.1f}s")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_params = {k: v.copy() for k, v in params.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no val improvement for {args.patience} epochs).")
                break

    # restore best parameters
    if best_params is not None:
        for k in params:
            params[k][...] = best_params[k]

    # ---- Save artifacts --------------------------------------------------- #
    np.savez(
        output_dir / "small_mlp_numpy.npz",
        W1=params["W1"], b1=params["b1"], W2=params["W2"], b2=params["b2"],
        target_mean_seconds=y_mean, target_std_seconds=y_std,
    )

    def metrics_on(x, y_sec):
        return evaluate_predictions(y_sec, model.forward(x) * y_std + y_mean)

    final_metrics = {"train": metrics_on(x_train, y_train_sec),
                     "validation": metrics_on(x_val, y_val_sec)}
    write_json(output_dir / "metrics.json", final_metrics)
    write_metrics_summary(output_dir / "metrics_summary.csv", "small_mlp_numpy", final_metrics)
    write_json(output_dir / "training_history.json", history)
    write_run_metadata(output_dir / "run_metadata.json", prep_dir, {
        "model_type": "small_numpy_mlp_regressor",
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": args.hidden_dim,
        "trainable_parameters": int(n_params),
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "epochs_run": len(history),
        "best_val_loss_std": best_val_loss,
        "target_mean_seconds": y_mean,
        "target_std_seconds": y_std,
        "seed": args.seed,
    })

    save_training_plot(history, output_dir / "training_curve.png", args.optimizer)

    print("\nSaved NumPy neural-network artifacts to:", output_dir)
    for split_name, m in final_metrics.items():
        print(f"  {split_name:<11} MAE={m['mae_seconds']:6.1f}s  RMSE={m['rmse_seconds']:6.1f}s  R2={m['r2']:.3f}")


def save_training_plot(history: list[dict], path: Path, optimizer_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_mae = [h["train_mae_seconds"] for h in history]
    val_mae = [h["val_mae_seconds"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_mae, label="train MAE", linewidth=1.8)
    ax.plot(epochs, val_mae, label="validation MAE", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title(f"TNP trip-duration MLP - training behavior ({optimizer_name}, NumPy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("Saved training-curve plot to:", path)


if __name__ == "__main__":
    main()
