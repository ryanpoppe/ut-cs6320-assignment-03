"""CS 6320 Assignment 3 - Part A: explicit neural-network training loop.

Goal of this script: make the *mechanics* of training visible rather than hiding
them behind a high-level ``.fit()`` call. Each epoch runs the five steps the
deep-learning frameworks do for you, written out and labeled in ``train_epoch``:

    1. forward pass        -> model(x) produces predictions
    2. loss computation    -> MSE between predictions and targets
    3. zero gradients      -> clear .grad from the previous step
    4. backpropagation     -> loss.backward() fills every parameter's .grad
    5. parameter update    -> step down the gradient (Adam, or hand-written SGD)

I keep the loop full-batch (no mini-batching, which the assignment says is
optional) so that one epoch == one forward/backward/update over the whole
training set, which is the clearest way to see the loop.

To show I understand what the optimizer is doing under the hood, ``--optimizer
manual_sgd`` replaces ``optimizer.step()`` with an explicit, hand-written
gradient-descent update inside ``torch.no_grad()``. The default is Adam.

The target (trip duration in seconds) is standardized with the training mean/std
for numerically stable MSE training; predictions are converted back to seconds so
that the reported MAE is interpretable as "typical error in seconds".

Run:
    python train_tnp_mlp.py
    python train_tnp_mlp.py --optimizer manual_sgd --learning-rate 0.05
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

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


class SmallMLP(nn.Module):
    """One hidden layer of ReLU units -> single regression output."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # shape (batch,) to match the targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", default=str(DEFAULT_PREP_DIR))
    parser.add_argument("--output-dir", help="Defaults to <prep-dir>/runs/small_mlp_torch/.")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=400, help="Full-batch passes over the data.")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "manual_sgd"],
        default="adam",
        help="adam = torch.optim.Adam; manual_sgd = hand-written p -= lr * p.grad update.",
    )
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def grad_global_norm(model: nn.Module) -> float:
    """L2 norm of all parameter gradients - evidence that backprop ran."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return math.sqrt(total)


def to_seconds(pred_std: torch.Tensor, y_mean: float, y_std: float) -> np.ndarray:
    """Invert the target standardization so errors are in seconds."""
    return (pred_std.detach().cpu().numpy() * y_std) + y_mean


def evaluate_seconds(
    model: nn.Module, x: torch.Tensor, y_seconds: np.ndarray, y_mean: float, y_std: float
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred_seconds = to_seconds(model(x), y_mean, y_std)
    return evaluate_predictions(y_seconds, pred_seconds)


def train_epoch(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train_std: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    learning_rate: float,
) -> tuple[float, float]:
    """One full-batch training step, with every stage of the loop made explicit.

    Returns (train_loss, gradient_norm).
    """
    model.train()

    # (1) FORWARD PASS: run the inputs through the network.
    predictions = model(x_train)

    # (2) LOSS: how wrong are the predictions (mean squared error)?
    loss = loss_fn(predictions, y_train_std)

    # (3) ZERO GRADS: clear gradients accumulated from the previous epoch.
    if optimizer is not None:
        optimizer.zero_grad()
    else:
        model.zero_grad()

    # (4) BACKPROP: autograd walks the graph and fills p.grad for every parameter.
    loss.backward()
    grad_norm = grad_global_norm(model)

    # (5) PARAMETER UPDATE: step the parameters downhill along their gradients.
    if optimizer is not None:
        optimizer.step()  # Adam handles the step internally
    else:
        # Hand-written stochastic gradient descent. This is exactly what
        # optimizer.step() does for plain SGD: w <- w - lr * dL/dw. Wrapped in
        # no_grad() so the update itself is not recorded by autograd.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    return float(loss.detach()), grad_norm


def main() -> None:
    args = parse_args()
    prep_dir = Path(args.prep_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else default_run_dir(prep_dir, "small_mlp_torch")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # ---- Data + preprocessing (shared with the baselines) ----------------- #
    split_pairs = split_xy(load_splits(prep_dir))
    x_train_df, y_train = split_pairs["train"]
    x_val_df, y_val = split_pairs["validation"]

    preprocessor = build_preprocessor()
    x_train_np = preprocessor.fit_transform(x_train_df).astype(np.float32)
    x_val_np = preprocessor.transform(x_val_df).astype(np.float32)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    y_train_seconds = y_train.to_numpy(dtype=np.float64)
    y_val_seconds = y_val.to_numpy(dtype=np.float64)

    # Standardize the target for stable MSE training; remember mean/std to invert.
    y_mean = float(y_train_seconds.mean())
    y_std = float(y_train_seconds.std())

    # ---- Tensors (full batch) --------------------------------------------- #
    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    y_train_std = torch.tensor((y_train_seconds - y_mean) / y_std, dtype=torch.float32)

    # ---- Model / loss / optimizer ----------------------------------------- #
    model = SmallMLP(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim)
    loss_fn = nn.MSELoss()
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.optimizer == "adam"
        else None  # manual_sgd does the update by hand in train_epoch
    )
    n_params = int(sum(p.numel() for p in model.parameters()))

    print(
        f"Model: SmallMLP(input_dim={x_train.shape[1]}, hidden_dim={args.hidden_dim}) "
        f"-> {n_params:,} trainable parameters"
    )
    print(f"Optimizer: {args.optimizer} (lr={args.learning_rate}) | loss: MSE on standardized target")
    print(f"Training full-batch for up to {args.epochs} epochs (patience={args.patience})\n")

    # ---- Training loop ---------------------------------------------------- #
    history: list[dict] = []
    best_state = None
    best_val_loss = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, grad_norm = train_epoch(
            model, x_train, y_train_std, loss_fn, optimizer, args.learning_rate
        )

        # VALIDATION TRACKING: standardized MSE on val, plus MAE in seconds for
        # both splits so the curve is interpretable.
        model.eval()
        with torch.no_grad():
            val_pred_std = model(x_val)
            val_loss = float(loss_fn(val_pred_std, torch.tensor((y_val_seconds - y_mean) / y_std, dtype=torch.float32)))
        train_metrics = evaluate_seconds(model, x_train, y_train_seconds, y_mean, y_std)
        val_metrics = evaluate_seconds(model, x_val, y_val_seconds, y_mean, y_std)

        history.append(
            {
                "epoch": epoch,
                "train_loss_std": train_loss,
                "val_loss_std": val_loss,
                "grad_norm": grad_norm,
                "train_mae_seconds": train_metrics["mae_seconds"],
                "val_mae_seconds": val_metrics["mae_seconds"],
            }
        )

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch {epoch:4d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"grad_norm {grad_norm:7.3f} | train_MAE {train_metrics['mae_seconds']:6.1f}s | "
                f"val_MAE {val_metrics['mae_seconds']:6.1f}s"
            )

        # Early stopping on validation loss.
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no val improvement for {args.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Save model + metrics + history ----------------------------------- #
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "hidden_dim": args.hidden_dim,
            "target_mean_seconds": y_mean,
            "target_std_seconds": y_std,
        },
        output_dir / "small_mlp.pt",
    )

    final_metrics = {
        "train": evaluate_seconds(model, x_train, y_train_seconds, y_mean, y_std),
        "validation": evaluate_seconds(model, x_val, y_val_seconds, y_mean, y_std),
    }
    write_json(output_dir / "metrics.json", final_metrics)
    write_metrics_summary(output_dir / "metrics_summary.csv", "small_mlp_torch", final_metrics)
    write_json(output_dir / "training_history.json", history)
    write_run_metadata(
        output_dir / "run_metadata.json",
        prep_dir,
        {
            "model_type": "small_pytorch_mlp_regressor",
            "input_dim": int(x_train.shape[1]),
            "hidden_dim": args.hidden_dim,
            "trainable_parameters": n_params,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "epochs_run": len(history),
            "best_val_loss_std": best_val_loss,
            "target_mean_seconds": y_mean,
            "target_std_seconds": y_std,
            "seed": args.seed,
        },
    )

    # ---- Training-behavior plot ------------------------------------------- #
    save_training_plot(history, output_dir / "training_curve.png", args.optimizer)

    print("\nSaved neural-network artifacts to:", output_dir)
    for split_name, m in final_metrics.items():
        print(
            f"  {split_name:<11} MAE={m['mae_seconds']:6.1f}s  "
            f"RMSE={m['rmse_seconds']:6.1f}s  R2={m['r2']:.3f}"
        )


def save_training_plot(history: list[dict], path: Path, optimizer_name: str) -> None:
    """Train vs. validation MAE (seconds) over epochs."""
    import matplotlib

    matplotlib.use("Agg")  # headless / CHPC-safe backend
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_mae = [h["train_mae_seconds"] for h in history]
    val_mae = [h["val_mae_seconds"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_mae, label="train MAE", linewidth=1.8)
    ax.plot(epochs, val_mae, label="validation MAE", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title(f"TNP trip-duration MLP - training behavior ({optimizer_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("Saved training-curve plot to:", path)


if __name__ == "__main__":
    main()
