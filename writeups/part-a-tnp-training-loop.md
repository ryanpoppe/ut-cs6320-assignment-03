# Ryan Poppe

# Assignment 3 — Part A: Neural-Network Training Loop on Chicago TNP Trip Duration

## Summary

I built a small neural network and an explicit, script-based training loop that predicts
`target_trip_seconds` (observed trip duration) for the Week 3 Chicago Transportation Network
Providers dataset. The loop exposes every mechanical step — forward pass, MSE loss,
gradient zeroing, backpropagation, parameter update, and per-epoch validation tracking — so
the point of the exercise is visible rather than hidden behind a high-level `fit()` call. On
a held-out validation split the network reaches **MAE ≈ 202 seconds (R² ≈ 0.77)**, beating a
mean-predictor floor (479 s) and a ridge-regression baseline (244 s) trained on the identical
preprocessed features. The whole pipeline runs from scripts and was executed on CHPC (SLURM
job 1450277, GPU node grn077); the captured job log (`slurm-tnp-1450277.out`), saved metrics,
and training-curve plot are included as evidence.

## Dataset and task

The data is the course-cleaned teaching subset `week03_tnp_trip_duration.csv` (30,000 public
Chicago TNP trip records). The task is **regression**: predict trip duration in seconds from
trip metadata. I used the eight features named in the Week 3 data dictionary and the documented
type guidance:

- **Numeric** (standardized): `trip_miles`, `start_hour`, `start_day_of_week`, `start_month`,
  `trips_pooled`.
- **Categorical** (one-hot, rare levels < 10 grouped): `pickup_community_area`,
  `dropoff_community_area`, `shared_trip_authorized`.

After preprocessing the model input dimension is **161**. The target ranges 61–7,123 s (mean
≈ 951 s). The fare/tip/charge fields were already excluded by the course to avoid
target-adjacent leakage. Consistent with the dataset notes, I use a single random
train/validation split (24,000 / 6,000 rows, seed 6320) and treat validation MAE as a
training diagnostic, **not** as a deployment claim — `trip_miles` is an observed (not pre-trip)
quantity, so this is a training-loop exercise, not an ETA system.

This task is intentionally *not* the Assignment 2 task. I reused my Assignment 2 *script
organization* (a shared `*_common.py` module, a separate data-prep step, separate train
scripts, saved metrics/log artifacts, the seed-6320 convention) but the dataset, the target,
and the problem type (regression vs. classification) are all new to Assignment 3.

## Model

A deliberately small feed-forward network (`SmallMLP`):

```
input (161) → Linear(161, 32) → ReLU → Linear(32, 1) → trip-duration prediction
```

One hidden layer of 32 ReLU units, **5,217 trainable parameters**. The target is standardized
with the training mean/std for numerically stable MSE training; predictions are converted back
to seconds before any metric is reported, so MAE reads directly as "typical error in seconds."

## Loss and optimizer

- **Loss:** mean squared error (`MSELoss`) on the standardized target. MSE is the standard
  regression loss and gives smooth gradients for backpropagation.
- **Optimizer:** Adam, learning rate 1e-2, default. The PyTorch script also supports a
  `--optimizer manual_sgd` mode that replaces `optimizer.step()` with a hand-written
  `w ← w − lr · ∂L/∂w` update, to show explicitly what the optimizer does.
- **Batching:** full-batch gradient descent (no mini-batching, which the assignment lists as
  optional). One epoch is one forward/backward/update over all 24,000 training rows, which is
  the clearest way to see the loop.
- **Early stopping:** training runs up to 400 epochs and stops when validation loss does not
  improve for 30 epochs, restoring the best-validation parameters.

## The training loop, step by step

Each epoch runs the five steps that a framework normally hides. In `train_tnp_mlp.py` they are
labeled exactly like this:

1. **Forward pass** — `predictions = model(x_train)` runs inputs through the network.
2. **Loss computation** — `loss = MSE(predictions, target)` measures how wrong they are.
3. **Zero gradients** — `optimizer.zero_grad()` clears `.grad` from the previous epoch (PyTorch
   accumulates gradients otherwise).
4. **Backpropagation** — `loss.backward()` runs autograd, filling every parameter's `.grad`
   with `∂L/∂parameter`. I log the global gradient L2 norm each epoch as direct evidence that
   backprop produced gradients (it falls from ≈0.53 at epoch 1 toward ≈0.01–0.09 as the model
   nears a flat region of the loss).
5. **Parameter update** — `optimizer.step()` steps each parameter downhill along its gradient
   (or the manual SGD update does it by hand).

**Validation tracking:** after each epoch I switch the model to eval mode, run a no-grad forward
pass on the validation split, and record standardized MSE plus MAE in seconds for both splits.
The full per-epoch history is saved to `training_history.json`.

To make the backpropagation concrete, I also wrote a dependency-free **NumPy twin**
(`train_tnp_mlp_numpy.py`) that implements the *same* network and loop but codes the backward
pass by hand — the chain rule layer by layer (`d_out → dW2, db2 → d_a1 → d_z1 (ReLU mask) →
dW1, db1`). This is exactly the calculation `loss.backward()` automates, written out. See the
reproducibility note below.

## Validation tracking — training behavior

Per-epoch behavior (CHPC PyTorch run; standardized loss and MAE in seconds):

| Epoch | Train loss | Val loss | Grad norm | Train MAE (s) | Val MAE (s) |
|------:|-----------:|---------:|----------:|--------------:|------------:|
| 1     | 0.9989     | 0.9560   | 0.530     | 465.9         | 471.7       |
| 25    | 0.3081     | 0.2959   | 0.310     | 234.7         | 235.2       |
| 50    | 0.2572     | 0.2625   | 0.088     | 217.0         | 220.8       |
| 75    | 0.2360     | 0.2498   | 0.024     | 206.5         | 214.2       |
| 100   | 0.2190     | 0.2392   | 0.019     | 197.6         | 208.0       |
| 125   | 0.2067     | 0.2350   | 0.016     | 191.3         | 205.4       |
| 150   | 0.1975     | 0.2329   | 0.013     | 186.4         | 204.2       |
| 175   | 0.1897     | 0.2317   | 0.031     | 183.0         | 203.5       |
| 200   | 0.1832     | 0.2312   | 0.088     | 178.8         | 202.1       |
| **206 (best)** | **0.1817** | **0.2308** | **0.087** | **178.6** | **202.4** |

Training ran 236 epochs and early-stopped, restoring epoch 206 (best validation loss). The
training-curve plot is saved at
`artifacts/tnp_prep/runs/small_mlp_torch/training_curve.png`: a steep early drop
(≈472 → ≈235 s by epoch 25), then a slow decline where train and validation separate slightly
— mild overfitting — until validation flattens near 202 s.

## Results — does the network beat the baselines?

Validation split (6,000 rows), all on the identical preprocessing:

| Model | Val MAE (s) | Val RMSE (s) | Val R² |
|---|---:|---:|---:|
| Mean predictor (floor) | 479.0 | 650.1 | −0.000 |
| Ridge regression | 243.7 | 358.0 | 0.697 |
| **Small neural network** | **202.4** | **309.6** | **0.773** |

(Network train metrics for reference: MAE 178.6 s, RMSE 274.4 s, R² 0.819.)

## Interpretation

The loop behaves the way the lecture mechanics predict. The loss decreases monotonically on
train, the gradient norm shrinks toward zero as the optimizer approaches a flat region, and
the network clears both baselines: it cuts the mean-predictor error by more than half and
improves on ridge by ≈41 s of MAE (≈17%). That gap is the part of the duration signal that is
genuinely **non-linear** in these features — plausibly interactions between distance, hour,
and pickup/dropoff area — which a linear model cannot capture but one hidden layer can.

The train/validation gap (179 s vs. 202 s) is mild, expected overfitting on a flexible model;
early stopping is what keeps it in check, and is the reason validation tracking is part of the
loop rather than an afterthought. Consistent with the assignment's scope, this is one small
model, one random split, a modest epoch budget, and no hyperparameter search — the goal was to
make the training mechanics visible and confirm the model learns real signal, not to chase the
last few seconds of MAE. I also keep the dataset's responsible-use framing: these numbers are
a training diagnostic on a teaching subset, not evidence of a deployable ETA model (notably,
`trip_miles` would not be known before a real trip).

## CHPC evidence

- **Job script:** `chpc/run_tnp.slurm` — submitted from the assignment-03 root with
  `sbatch chpc/run_tnp.slurm`. It reuses the Week 3 dc_energy template allocation
  (`account=larsenc`, `partition`/`qos=utucset-gpu-grn`, `gres=gpu:1`), builds a venv from
  `requirements.txt`, and runs the pipeline.
- **Executed run:** SLURM **job 1450277** on GPU node **grn077** (Python 3.12.4, NVIDIA RTX PRO
  6000 GPU), 2026-06-17. Captured stdout is **`slurm-tnp-1450277.out`** (stderr in
  `slurm-tnp-1450277.err`) — the primary CHPC evidence.
- **Command sequence** (also in the job script and README):
  `python prepare_tnp_data.py` → `python train_baseline.py` → `python train_tnp_mlp.py`.
- **Saved outputs (CHPC):** `artifacts/tnp_prep/manifest.json`, the prepared splits under
  `artifacts/tnp_prep/data/`, and per-model `metrics.json`, `metrics_summary.csv`,
  `run_metadata.json`, the saved model (`small_mlp.pt`), `training_history.json`, and
  `training_curve.png` under `artifacts/tnp_prep/runs/small_mlp_torch/` (baselines under
  `baseline_mean/` and `baseline_ridge/`).

## Reproducibility note — PyTorch on CHPC vs. the NumPy twin

The submitted training-loop implementation is **`train_tnp_mlp.py` (PyTorch / autograd)**, and
it is what produced the headline numbers above when run on CHPC (job 1450277). During offline
development I did not have a PyTorch install, so I built an **equivalent NumPy twin**
(`train_tnp_mlp_numpy.py`) that implements the identical architecture, target standardization,
full-batch loop, optimizer, and early stopping, with the backward pass written out by hand —
exactly what autograd computes automatically.

The two agree closely, which is the point. The deterministic baselines are identical to many
decimals (mean MAE 479.0 s, ridge MAE 243.7 s in both runs). The networks land in the same
place: CHPC PyTorch reaches **val MAE 202.4 s (R² 0.773)** and the laptop NumPy twin reaches
**val MAE 209.0 s (R² 0.767)**. The ≈7-second difference comes from PyTorch's default weight
initialization and Adam details versus the NumPy He-initialization, which send the optimizer
down slightly different paths (the PyTorch run trained 236 epochs vs. 167 and reached a marginally
lower best validation loss). The CHPC PyTorch outputs live under `artifacts/`; the NumPy-twin
outputs from the laptop run are kept under `artifacts_local/` for comparison.

## Sources

- Chicago Transportation Network Providers — Trips (2018–2022), City of Chicago:
  https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p/about_data
- Week 3 course materials: `week03_tnp_trip_duration.csv`, `data_dictionary.md`,
  `dataset_audit.md`, `dataset_notes.md`, and the `dc_energy` example scripts + `run.slurm`
  CHPC template.
