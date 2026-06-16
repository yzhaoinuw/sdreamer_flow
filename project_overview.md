# Project Overview

This document orients a new agent (or human collaborator) to the codebase. It's the single most valuable artifact for getting useful work done quickly. Keep it current — when the active code path changes, update the relevant sections here.

## What This Repo Is

`sdreamer_flow` is the training pipeline for **sDREAMER** — a *self-distilled Mixture-of-Modality-Experts (MoME) transformer* for automatic mouse sleep staging from paired **EEG + EMG**. The model reads fixed-length epochs (512 samples each, ~1 s after resampling) and classifies each into one of three sleep stages: **Wake (0), NREM/SWS (1), REM (2)**. It is the model behind the lab's sleep-scoring tooling; see the [sDREAMER paper](https://www.cs.rochester.edu/u/yyao39/files/sDREAMER.pdf) cited in `README.md`.

This repo is a trimmed "flow" subset of a larger research codebase. The original explored many architectures (ViT, CM/CMA, Macaron, Freq/STFT, LSTM, several MoE variants, with/without an NE neuromodulation channel) — most of that still exists in-tree as **legacy reference** but is *not* on the active path. The live pipeline trains a single sequence MoME model.

End-to-end flow:

1. **Data prep** — `.mat` recordings → standardized, sliced train/val `.npy` tensors (`write_training_data.py`).
2. **Training** — load those tensors and train the transformer (`run_train_sdreamer.py` → `exp/exp_moe2.py` → `models/seq/n2nSeqNewMoE2.py`).

Stack: PyTorch, `einops`, `timm` (DropPath / weight init), scikit-learn (KFold + metrics), scipy (`.mat` I/O + resampling). PyTorch Lightning is imported only for `seed_everything`.

## Active Runtime Path

### 1. Data-prep entrypoint

[`write_training_data.py`](write_training_data.py)

- `reshape_sleep_data` (in [`utils/preprocessing.py`](utils/preprocessing.py)) flattens EEG/EMG, trims trailing missing labels, resamples to **512 samples per epoch** (`scipy.signal.resample_poly` when the sampling rate ≠ 512), and reshapes into per-second epochs.
- `prepare_data` standardizes EEG and EMG independently, stacks them into `[N, trace=2, channel=1, 512]`, then `slice_data` groups epochs into sequences of `seq_len=64`. The leftover tail is kept by appending a final full window of 64 (so no epochs are dropped).
- `augment=True` upsamples windows around **REM (class 2)** transitions (`upsampling_scale` copies each) to fight REM rarity.
- 5-fold split via `KFold(n_splits=5, shuffle=True, random_state=42)`; writes `train_trace{fold}.npy`, `train_label{fold}.npy`, `val_trace{fold}.npy`, `val_label{fold}.npy` plus a `train_val_split.txt` manifest into `save_path` (default `.../sdreamer_data/n_seq_64/fold_1/`). Files in `on_hold_list`, or any file with NaN / `-1` labels, are excluded.

### 2. Training entrypoint

[`run_train_sdreamer.py`](run_train_sdreamer.py)

- Holds the full **hyperparameter `config` dict** (the single source of truth for a run — there is no CLI; `argparse` is created but no flags are added, so `config` is copied onto `args`).
- Sets all RNG seeds (`seed=42`), builds the `setting` string used for the checkpoint dir / log name, and attaches a file logger at `<checkpoints>/<setting>.log`.
- Paths set in `__main__`: `data_path="../sdreamer_data/"`, `checkpoints="../sdreamer_checkpoints/"`, `des_name="test"`.
- Instantiates `Exp_MoE(args)` and calls `exp.run_train(setting)`.

### 3. Experiment driver

[`exp/exp_moe2.py`](exp/exp_moe2.py) — class `Exp_MoE`

- `_build_model`: only `SeqNewMoE2` and `SeqHMoE` are registered; **every other model is commented out**. With `features="ALL"` it builds the two-trace `Model`; otherwise `Mono_Model`.
- `_get_data` → `data_provider/data_generator.py` → `Seq_Loader`.
- `_select_criterion`: `CrossEntropyLoss` (+ unused `CosineEmbeddingLoss`, `KLDivLoss` for distillation).
- `train` / `eval`: standard loop with `utils/metrics.ProgressMeter` + `utils/metric_tracker.build_tracker_mome` (acc, macro-F1, precision, recall, Cohen's κ for the mixed head and the eeg/emg heads).
- **Loss** = `loss1 + (distill_eeg + distill_emg) * scale`, where `loss1` is CE on the mixed head and the distill terms are `KLDiv` from the EEG/EMG single-modality heads toward the mixed head (temperature ~2.0 on the EEG term). **At the default `scale=0.0` this is just plain CE on the mixed head.**
- `run_train`: AdamW + CosineAnnealingLR, `EarlyStopping` on **val accuracy** (`patience=30`), checkpoints (`ckpt.pth.tar` + `model_best.pth.tar`) via `utils/tools.py`.

### 4. Model architecture

[`models/seq/n2nSeqNewMoE2.py`](models/seq/n2nSeqNewMoE2.py) — class `Model`. Layers live in [`layers/`](layers).

Input batch: `[batch, n_sequences=64, trace=2, channel=1, time=512]`. `x[:, :, 0]` is EEG, `x[:, :, 1]` is EMG.

1. **Per-epoch encoders** — two independent `Transformer`s (`layers/transformer.py`), one for EEG and one for EMG. Each: `PatchEncoder` splits the 512-sample epoch into `patch_len=16` → **32 patches** (Linear→ReLU→Linear to `d_model=128`), prepends a CLS token, adds learned positional embeddings, then `e_layers=2` pre-norm `MultiHeadAttention` blocks (GLU FFN, LayerScale γ, DropPath). The **per-epoch CLS token** (`[:, :, -1]`) summarizes each epoch.
2. **Sequence MoME** — the 64 EEG-CLS and 64 EMG-CLS vectors feed `SeqNewMoETransformer2` (`layers/transformer.py`), built from `MoEBlock`s (`layers/attention.py`). Shared self-attention across the concatenated EEG+EMG token stream; the **FFN is the "expert"**: a separate `mlp_eeg` / `mlp_emg` for modality-specific layers, and a shared `mlp_mix` once `layer_index >= mixffn_start_layer_index`. With `seq_layers=3, ca_layers=1` → `mixffn_start_layer_index = 2`, so layers 0–1 route per-modality and layer 2 mixes. A learned modality embedding (`nn.Embedding(2, …)`) tags EEG vs EMG.
3. **Three heads** — `infer` (mixed, via `SeqPooler2` which concatenates the EEG/EMG halves), `infer_eeg`, `infer_emg`. Each `cls_head` is `Linear→LayerNorm→GELU→Linear` to `c_out=3`. Outputs are flattened to `[batch*n_sequences, 3]`. The eeg/emg heads exist so the mixed head can self-distill into them (when `scale>0`).

The second registered model, [`models/seq/n2nSeqHMoE.py`](models/seq/n2nSeqHMoE.py), is a hierarchical-MoE variant kept available through the same `Exp_MoE`.

### Key hyperparameters (defaults in `run_train_sdreamer.py`)

| Param | Value | Meaning |
|---|---|---|
| `model` / `data` | `SeqNewMoE2` / `Seq` | active model + sequence loader |
| `features` | `ALL` | use both EEG and EMG (vs mono) |
| `c_out` | `3` | Wake / NREM / REM |
| `seq_len` | `512` | samples per epoch |
| `n_sequences` | `64` | epochs per training sequence |
| `patch_len` / `stride` | `16` / `8` | patching of the 512-sample epoch (→ 32 patches) |
| `d_model` / `n_heads` | `128` / `8` | transformer width / heads |
| `d_ff` | `512` | FFN inner dim (`mult_ff = d_ff/d_model = 4`) |
| `e_layers` | `2` | per-epoch encoder depth |
| `seq_layers` / `ca_layers` | `3` / `1` | sequence MoME depth / mixed-FFN tail layers |
| `mix_type` | `0` | positional emb on, modality emb off (in the per-epoch encoders) |
| `activation` / `norm_type` | `glu` / `layernorm` | FFN activation / norm |
| `dropout` / `path_drop` | `0.1` / `0.0` | dropout / stochastic depth |
| `scale` | `0.0` | self-distillation weight (0 ⇒ plain CE) |
| `optimizer` | `adamw`, `lr=1e-3`, `weight_decay=1e-4` | (note: AdamW path ignores `beta`/`eps` args) |
| `scheduler` | `CosineLR` | `CosineAnnealingLR` over `epochs` |
| `epochs` / `patience` | `100` / `30` | max epochs / early-stop patience on val acc |
| `batch_size` | `64` | |
| `fold` | `1` | which of the 5 folds is held out for val |
| `useNorm` | `True` | use the standardized trace channel |

## Repo Structure Map

```text
sdreamer_flow/
|- AGENTS.md                  # treaty: runtime, tasks, gotchas (read first)
|- project_overview.md        # this file
|- next_steps.md              # in-flight threads
|- work_log.md                # session history (+ work_log_archive/)
|- README.md                  # paper citation + ORIGINAL (stale) file-structure map
|- run_train_sdreamer.py      # ACTIVE training entrypoint + hyperparameter config
|- write_training_data.py     # ACTIVE data prep (.mat -> .npy)
|- exp/                       # experiment drivers; exp_moe2.py is ACTIVE, rest legacy
|- models/                    # epoch/ + seq/ model zoo; seq/n2nSeqNewMoE2.py is ACTIVE
|- layers/                    # transformer / attention / patch / head / norm building blocks
|- data_provider/             # data_generator.py + data_loader.py (Seq_Loader is ACTIVE)
|- utils/                     # preprocessing, optimization, tools, metrics, visualize
|- scripts/, scripts_ne/      # legacy shell-script experiment launchers
|- data/                      # only data/dst_data_wNE/ stub dirs remain here
|- logs/, ml_base*.ipynb, test.png   # legacy artifacts
|- moe_Launch*.py, train_Launch*.py, moe_Eval.py   # legacy launchers (superseded)
```

> Note: the actual training data and checkpoints live **outside** this repo, in sibling dirs
> `../sdreamer_data/`, `../sdreamer_checkpoints/`, `../sdreamer_input_data/`, `../sdreamer_output_data*/`.

## What Looks Active vs. Legacy

This is the single most important section for agents — the repo carries many parallel implementations.

### Active / relevant now

- [`run_train_sdreamer.py`](run_train_sdreamer.py) — training entrypoint + the one real config.
- [`write_training_data.py`](write_training_data.py) — data prep.
- [`exp/exp_moe2.py`](exp/exp_moe2.py) — the experiment driver (`Exp_MoE`).
- [`models/seq/n2nSeqNewMoE2.py`](models/seq/n2nSeqNewMoE2.py) — the trained model (`SeqNewMoE2`).
- [`models/seq/n2nSeqHMoE.py`](models/seq/n2nSeqHMoE.py) — alternative `SeqHMoE`, also registered.
- [`layers/transformer.py`](layers/transformer.py), [`layers/attention.py`](layers/attention.py), [`layers/patchEncoder.py`](layers/patchEncoder.py), [`layers/head.py`](layers/head.py), [`layers/norm.py`](layers/norm.py) — building blocks used by the active model.
- [`data_provider/data_generator.py`](data_provider/data_generator.py) + [`data_provider/data_loader.py`](data_provider/data_loader.py) — `Seq_Loader` is the active dataset.
- [`utils/preprocessing.py`](utils/preprocessing.py), [`utils/optimization.py`](utils/optimization.py), [`utils/tools.py`](utils/tools.py), [`utils/metrics.py`](utils/metrics.py), [`utils/metric_tracker.py`](utils/metric_tracker.py).

### Likely older or secondary

- `exp/exp_main.py`, `exp/exp_moe.py`, `exp/exp_ne.py`, `exp/exp_moe_ne.py`, `exp/exp_moe2_crf.py` — earlier / NE / CRF experiment drivers, not imported by the active entrypoint.
- All of `models/epoch/*` and the other `models/seq/*` (ViT, CM/CMA, Macaron, Macross, Freq/TF, LSTM, MLP, BaseLine, the `*NE` and `*_crf` variants) — the broader research zoo; reference only.
- `moe_Launch.py`, `moe_Launch2.py`, `moe_LaunchNE.py`, `train_Launch.py`, `train_LaunchNE.py`, `moe_Eval.py` — older launch/eval scripts superseded by `run_train_sdreamer.py`.
- `scripts/`, `scripts_ne/` — shell launchers for the old experiment grid.
- `data_provider/data_generator_ne.py`, `data_loader_testfile.py`, `dis_plot.py.save`, `.swp` files — NE / scratch artifacts.
- `ml_base.ipynb`, `ml_baseline.ipynb`, `logs/`, `test.png` — exploratory ML baselines / leftovers.
- The **`README.md` file-structure block describes the *original* sDREAMER repo by its authors**, kept intentionally as a record of that upstream layout. It does not describe this trimmed repo (it lists `ckpt/`, `ckpt_ne/`, `ckpt_seq/`, `visualizations/`, `epoch_pics/`, etc. that aren't here) — so trust *this* file for the current structure, but leave that block in place.

## Tests And Fixtures

There is no automated test suite (the Copier `test_command` is empty). Verification is manual:

- Import smoke check on the active modules: `python -c "from exp.exp_moe2 import Exp_MoE; from models.seq import n2nSeqNewMoE2, n2nSeqHMoE"`.
- Data-prep sanity: confirm `write_training_data.py` writes the four `.npy` files + `train_val_split.txt` and that shapes are `[N, 64, 2, 1, 512]` (traces) / `[N, 64, 1]` (labels).
- Training sanity: run a few epochs and watch the `<setting>.log` for rising train/val accuracy.

Canonical sample data: the lab's preprocessed `.mat` recordings (see the `data_path` in `write_training_data.py`'s `__main__`) and the cached tensors under `../sdreamer_data/n_seq_64/fold_1/`.

## User Data Expectations

`write_training_data.py` consumes `.mat` files via `reshape_sleep_data`. Each `.mat` must contain:

- `eeg` — 1-D EEG samples.
- `emg` — 1-D EMG samples.
- `eeg_frequency` — scalar sampling rate (Hz); used to resample to 512 samples/epoch.
- `sleep_scores` — per-epoch integer labels: **0 = Wake, 1 = NREM/SWS, 2 = REM**; `-1` / NaN mark missing/unscored epochs.

Files whose labels contain any NaN or `-1` are skipped entirely during data writing. Trailing missing labels are trimmed; EEG/EMG and labels are truncated to the shorter length.

The cached training tensors (`Seq_Loader`) are `traces [N, n_sequences, 2, channel, 512]` and `labels [N, n_sequences, 1]`. The data was originally stored with both raw and normalized channels stacked; `useNorm` selects which channel slice is used (the active default uses the standardized one).

## Practical Mental Model

If you only want to understand the current pipeline, read in this order:

1. [`README.md`](README.md) — paper context (ignore its stale file-structure block).
2. [`run_train_sdreamer.py`](run_train_sdreamer.py) — config + entrypoint.
3. [`write_training_data.py`](write_training_data.py) + [`utils/preprocessing.py`](utils/preprocessing.py) — how data is made.
4. [`exp/exp_moe2.py`](exp/exp_moe2.py) — the training loop + loss.
5. [`models/seq/n2nSeqNewMoE2.py`](models/seq/n2nSeqNewMoE2.py) — the model.
6. [`layers/transformer.py`](layers/transformer.py) + [`layers/attention.py`](layers/attention.py) — the MoME internals.

## Questions Worth Clarifying Later

Not blockers, just places where maintainer intent would help future agents:

- Should the legacy model zoo (`models/epoch/*`, non-active `models/seq/*`, old launchers, `scripts*/`) be pruned from this "flow" repo, or kept as reference?
- Is the CRF line of work (`exp/exp_moe2_crf.py`, `models/seq/n2nSeqNewMoE2_crf.py`) abandoned? Recent commit `c418786` reverted "moe crf back to just moe."
- Should `run_train_sdreamer.py` keep an all-in-file `config` dict, or move to real CLI flags? (The `argparse` parser currently defines no arguments.)
- Is the `NE` (neuromodulation channel) branch in scope for this repo or fully parked?
- The unweighted-CE footgun (`weight=[1,1,1]` ⇒ `None`) — is unweighted loss intentional given REM class imbalance, or should weighting be wired up properly?
