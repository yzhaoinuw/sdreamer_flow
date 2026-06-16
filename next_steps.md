# Next Steps

Use this checklist alongside `work_log.md`.

## Currently Hot

Active threads — read these first to know what work is in flight:

- **NE experiment port (branch `experiments/ne`)** — porting the intern's (Jaysen) best-performing work: a single model that scores with **or** without NE, trained by zeroing the NE channel for half of each batch ("mixed" training, his val acc ~0.928). Code is ported but **untested**: `utils/preprocessing_ne.py`, `write_training_data_ne.py`, `run_train_ne.py`, and a gated `ne_mix_ratio` in `exp/exp_ne.py`. See `EXPERIMENTS.md` for findings + the run/verify TODO (needs NE-bearing `.mat` data + a GPU run). Related future branches: `experiments/piecewise-norm`, `experiments/augmentation`.
- **Data prep on new Mie dataset** — a `write_training_data.py` edit (repoint to the lab's `Mie_newdata/preprocessed_data/`, write to `../sdreamer_data/n_seq_64/fold_1/`, `augment=True`) is **parked in `git stash` (`stash@{0}`)**, not in the working tree. It was based on the old pre-treaty `main` (`c418786`); the committed `write_training_data.py` has since been reformatted upstream, so reapply the change by hand rather than `git stash pop` blindly. Next action: reapply the path/augment change, regenerate the `.npy` tensors, confirm shapes.

Other sections below are background or paused; treat them as reference unless a new request reopens them.

## Data prep on new Mie dataset (claude-opus-4-8, default thinking)

Status: parked in `git stash` (`stash@{0}`); needs reapplying onto the reformatted `write_training_data.py`

The edit makes the `__main__` block use absolute lab paths instead of the old Windows `C:/Users/...` paths and turns on REM augmentation (`augment=True`), matching the default `data_path="../sdreamer_data/"` that `run_train_sdreamer.py` reads. ⚠️ It currently lives in `git stash` (`stash@{0}`), based on the pre-treaty `main` (`c418786`); since then the committed `write_training_data.py` was reformatted (seq_len/fold reordered, the `train_val_split.txt` write reflowed), so reapply the change by hand instead of popping the stash blindly.

Remaining work:

- Run `python write_training_data.py`, verify the four `.npy` files + `train_val_split.txt` land under `../sdreamer_data/n_seq_64/fold_1/` with expected shapes (`[N, 64, 2, 1, 512]` traces, `[N, 64, 1]` labels).
- Kick off a short `run_train_sdreamer.py` run and confirm train/val accuracy looks sane in the `<setting>.log`.
- Reapply the stashed path + `augment=True` change onto the reformatted file, then commit it.

## Background / Paused

Sections below this line are older threads kept for context.

### CRF top-layer experiment

Last-known state: an experiment to add a CRF as the top layer (`exp/exp_moe2_crf.py`, `models/seq/n2nSeqNewMoE2_crf.py`) was built out (commits `5b67bcf`…`2fe5828`) then reverted on the active path — commit `c418786` "change moe crf back to just moe." The CRF files remain in-tree but are not imported by the active entrypoint. Would be resumed only if sequence-label smoothing via CRF is revisited.

### NE (neuromodulation) branch

Last-known state: a parallel pipeline that adds an NE channel exists (`*NE.py` models, `data_generator_ne.py`, `exp_ne.py` / `exp_moe_ne.py`, `scripts_ne/`, `data/*_wNE/`). Not on the active path; parked pending a decision on whether NE data is in scope for this repo.
