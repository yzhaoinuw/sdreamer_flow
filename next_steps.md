# Next Steps

Use this checklist alongside `work_log.md`.

## Currently Hot

Active threads — read these first to know what work is in flight:

- **Data prep on new Mie dataset** — `write_training_data.py` is edited (uncommitted) to point at the lab's `Mie_newdata/preprocessed_data/` and write to `../sdreamer_data/n_seq_64/fold_1/` with `augment=True`. Next action: regenerate the `.npy` tensors and confirm shapes, then commit the path/augment change.

Other sections below are background or paused; treat them as reference unless a new request reopens them.

## Data prep on new Mie dataset (claude-opus-4-8, default thinking)

Status: in progress (working-tree change, not committed)

The `__main__` block of `write_training_data.py` now uses absolute lab paths instead of the old Windows `C:/Users/...` paths, and turns on REM augmentation (`augment=True`). This matches the default `data_path="../sdreamer_data/"` that `run_train_sdreamer.py` reads.

Remaining work:

- Run `python write_training_data.py`, verify the four `.npy` files + `train_val_split.txt` land under `../sdreamer_data/n_seq_64/fold_1/` with expected shapes (`[N, 64, 2, 1, 512]` traces, `[N, 64, 1]` labels).
- Kick off a short `run_train_sdreamer.py` run and confirm train/val accuracy looks sane in the `<setting>.log`.
- Commit the data-prep path + `augment=True` change.

## Background / Paused

Sections below this line are older threads kept for context.

### CRF top-layer experiment

Last-known state: an experiment to add a CRF as the top layer (`exp/exp_moe2_crf.py`, `models/seq/n2nSeqNewMoE2_crf.py`) was built out (commits `5b67bcf`…`2fe5828`) then reverted on the active path — commit `c418786` "change moe crf back to just moe." The CRF files remain in-tree but are not imported by the active entrypoint. Would be resumed only if sequence-label smoothing via CRF is revisited.

### NE (neuromodulation) branch

Last-known state: a parallel pipeline that adds an NE channel exists (`*NE.py` models, `data_generator_ne.py`, `exp_ne.py` / `exp_moe_ne.py`, `scripts_ne/`, `data/*_wNE/`). Not on the active path; parked pending a decision on whether NE data is in scope for this repo.
