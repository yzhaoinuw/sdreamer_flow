# Experiments

Performance experiments that are **not** on the active training path (`run_train_sdreamer.py`).
Each lives on an `experiments/*` branch. This file tracks findings and what's been ported.

## Source: intern (Jaysen / Zhisen Cong) experiments, Sep–Dec 2024

Repo: `../../jaysenSS/sdreamer_train_jsc/` (his GitHub: `zhisencong727/sdreamer_train_jsc`).
It's a **copy** of `sdreamer_flow` (fresh "Initial commit", all commits by `zcong2@u.rochester.edu`),
not a fork of our history. His handoff note is `difference.txt` in that repo. His source data
(`groundtruth_data/`, 68 `.mat`) is the **same Mie recordings** plus an added **NE channel**.
His checkpoints + training logs are under `trainedModel/` (each completed run has `model_best.pth.tar`).

### Best validation results (his runs; early-stopped on val acc)

| run | model | val Acc | F1 | κ | note |
|---|---|---|---|---|---|
| `ne_mixed_half` | BaseLineNE | **0.928** | 0.908 | 0.864 | best accuracy |
| `ne_mixed_50_masked` | BaseLineNE | 0.926 | **0.9095** | 0.859 | best F1 |
| `ne_sample` | BaseLineNE | 0.927 | 0.907 | 0.862 | |
| `ne_mixed_50` | BaseLineNE | 0.925 | 0.904 | 0.858 | |
| `ne_augmented` | BaseLineNE | 0.922 | 0.889 | 0.853 | REM augmentation **lower** |
| `ne_mixed_50_patch_len=8` | BaseLineNE | 0.917 | 0.889 | 0.843 | pl8 **worse** than pl16 |
| `ne_256` | BaseLineNE | 0.914 | 0.880 | 0.836 | ns256 **worse** than ns64; ns512 didn't finish |
| `Piece-wise_1` | SeqNewMoE2 | 0.911 | 0.875 | 0.832 | piecewise EMG norm |
| `piecewise_augment` | SeqNewMoE2 | **0.851** | 0.823 | 0.760 | piecewise + REM augmentation **crashed** |

### Findings

1. **NE channel + "mixed" training is his strongest result** (~0.925–0.928). The mechanism (in his
   `exp_ne.py`): zero the NE channel for ~50% of each batch so **one model learns to score with and
   without NE**. He confirmed he made **no model/transformer changes** — `layers/ne_moe.py` (the
   3-modality EEG+EMG+NE MoME) was already in our repo. So this is a **training-process** change.
2. **REM augmentation (upsampling) hurts** in both threads — piecewise 0.911 → 0.851 with augment;
   NE augmented 0.922 < mixed 0.925–0.928. Causes REM over-prediction. ⚠️ Relevant to the parked
   `augment=True` change on the Mie data-prep thread (see `next_steps.md`).
3. **Hyperparameters confirmed:** `patch_len=16` > 8; `n_sequences=64` > 256/512.
4. **Piecewise 64-s EMG z-score** (range-shift fix): his note reports "slight improvement" but
   REM over-prediction. Marginal — kept for a separate `experiments/piecewise-norm` branch.

Caveats: his val split differs slightly from ours (he excludes 1 file vs our 4 → different KFold),
so don't compare his absolute numbers to our baseline — only within his runs. His `exp_ne` also
zeroed NE in the **val** loop, so his reported acc is on a 50%-NE-zeroed val set. Logs have no
per-class breakdown, so "REM over-prediction" is his qualitative call.

## Ported into this branch (`experiments/ne`)

Code-only, **untested** (no GPU / NE data wired here):

- `utils/preprocessing_ne.py` — reads `ne` + `ne_frequency`, returns eeg/emg/ne/labels (copied as-is).
- `write_training_data_ne.py` — builds NE train/val tensors (`train_ne{fold}.npy` etc.). Paths
  adapted: input → Jaysen's `groundtruth_data` (NE-bearing `.mat`), output → `../sdreamer_data_ne/`.
- `run_train_ne.py` — NE training entrypoint (model `BaseLine`/`BaselineNE`, `ne_patch_len=10`).
  Paths adapted; `n_sequences=64`; added `ne_mix_ratio=0.5`.
- `exp/exp_ne.py` — added the **mixed-training** NE-zeroing in the train + val loops, **gated on
  `ne_mix_ratio`** (default `0.0` → no behavior change; `run_train_ne.py` sets `0.5`).

Not ported (incidental fixes he needed in his env — apply if you hit them): `label = label.long()`
casts and disabling `visualize_data` in `exp_ne`.

## To actually run / verify (TODO)

1. Confirm the NE `.mat` source has `eeg/emg/ne/ne_frequency/sleep_scores` (his `groundtruth_data` does).
2. `python write_training_data_ne.py` → writes tensors to `../sdreamer_data_ne/n_seq_64/fold_1/`.
3. `python run_train_ne.py` on a GPU node; watch `<setting>.log` for val acc (target ~0.92+).
4. If dtype/visualize errors appear, apply the two incidental fixes noted above.

## Future experiment branches
- `experiments/piecewise-norm` — windowed EMG z-score (finding #4).
- `experiments/augmentation` — controlled augment on/off + lower `upsampling_scale` (finding #2).
