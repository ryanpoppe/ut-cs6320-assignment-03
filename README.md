# CS 6320 Assignment 3 — Ryan Poppe

Two labeled parts:

- **Part A — Neural-network training loop** (Chicago TNP trip-duration regression, run on CHPC).
- **Part B — Portfolio project proposal** (NOAA ENSO seasonal forecasting + dataset pre-audit).

## Layout

```
ut-cs6320-assignment-03/
├── datasets/student_visible/week03_tnp_trips/
│   └── week03_tnp_trip_duration.csv     # Week 3 dataset
├── tnp_common.py                        # shared: load, split, preprocess, metrics, writers
├── prepare_tnp_data.py                  # train/validation split + manifest
├── train_baseline.py                    # mean predictor + ridge regression (context)
├── train_tnp_mlp.py                     # SUBMITTED training loop (PyTorch, autograd) — CHPC
├── train_tnp_mlp_numpy.py               # equivalent NumPy twin (hand-coded backprop) — evidence
├── chpc/run_tnp.slurm                   # CHPC SLURM job script
├── requirements.txt
├── artifacts/
│   ├── run_logs/tnp_run.log             # captured end-to-end run (evidence)
│   └── tnp_prep/...                      # prepared splits + per-model metrics/model/history/plot
└── writeups/
    ├── part-a-tnp-training-loop.md
    └── part-b-portfolio-proposal.md
```

## Run on CHPC (Part A)

```bash
cd ut-cs6320-assignment-03
sbatch chpc/run_tnp.slurm        # uses dc_energy allocation; captures slurm-tnp-<jobid>.out
```

The job runs the pipeline below and saves outputs under `artifacts/tnp_prep/runs/`.

## Run locally (same pipeline)

```bash
pip install -r requirements.txt
python prepare_tnp_data.py                 # -> artifacts/tnp_prep/{data,manifest.json}
python train_baseline.py                   # mean + ridge baselines
python train_tnp_mlp.py                    # PyTorch training loop (needs torch)
# or, with no torch installed, the equivalent NumPy twin:
python train_tnp_mlp_numpy.py
```

Useful flags on the training scripts: `--epochs`, `--learning-rate`, `--hidden-dim`,
`--patience`, and `--optimizer` (`adam` default; `manual_sgd` in the torch script / `sgd` in
the NumPy twin) to expose the hand-written parameter update.

## Note on evidence

The submitted training-loop implementation is `train_tnp_mlp.py` (PyTorch / autograd). It was
run on CHPC as **SLURM job 1450277** (GPU node grn077); the captured stdout is
`slurm-tnp-1450277.out` and the resulting metrics/model/history/plot are under
`artifacts/tnp_prep/runs/small_mlp_torch/`. The equivalent NumPy twin
(`train_tnp_mlp_numpy.py`) was used during offline development; its laptop outputs are kept
under `artifacts_local/` for comparison. The two implement the same model and loop and agree
closely (CHPC PyTorch val MAE 202.4 s vs. NumPy twin 209.0 s; identical baselines). See the
reproducibility note in the Part A writeup.

## Reused from Assignment 2

Script organization (shared `*_common.py`, separate prep/train scripts, saved
metrics/log/metadata artifacts), the train/validation + scaling/encoding patterns, the small
single-hidden-layer MLP and results-table style, the seed-6320 convention, and the SLURM/CHPC
job-script habit. The dataset, target, and task (regression vs. Assignment 2's classification)
are new, as required.
