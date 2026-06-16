# Work Log

Prepend new session notes to the top of this file.

Rotation policy: the live log holds at most the **5 most recent unique calendar dates**. When a new date would push the file past 5 unique dates, move the oldest 5 dates as a chunk into a new file at `work_log_archive/work_log_<earliest>_to_<latest>.md`. The live file always holds at most 5 unique dates; each archive file always holds exactly 5.

If today's date already has a `## YYYY-MM-DD` header at the top, add a new `###` session subsection under it rather than starting a second `## YYYY-MM-DD` header for the same date.

Update this log at the end of any substantive work session unless the user explicitly asks not to document it. Substantive work includes file edits, meaningful validation or debugging, technical decisions or reversals, reusable discoveries, branch/PR/release state changes, or follow-up work that future agents need. Log useful experiments even when the code was reverted; skip casual Q&A, trivial one-off commands, and pure scratch work with no future coordination value.

<!--
Each session entry follows this shape:

## YYYY-MM-DD

### Short title for what was done (model + version, effort/thinking mode, token budget if known)

- bullet describing what was added or changed
- another bullet — keep them high-level and user/agent-facing, not implementation play-by-play
- if relevant, intended profiling signal or measurement:
  - what to look for in logs / output
  - what numbers were observed
- Verification:
  - the exact command(s) that were actually run
  - what passed / what was confirmed

Model / effort / token info goes in the parentheses after the `###` title when available from the system. Use whatever the model or interface actually reports — do not estimate or hallucinate. Omit any field that the interface does not surface.

- **Model**: the version string the interface reports (e.g. `grok-4.3`, `gpt-4o`, `claude-opus-4-7`).
- **Effort / thinking mode**: the effort knob the interface reports (e.g. `high`, `low`, `extended thinking`). Omit if no such knob exists or its setting is not surfaced.
- **Token budget**: **output tokens for the session** (output + thinking/reasoning tokens for models that report them separately, e.g. Claude with extended thinking). This is the cleanest cross-agent proxy for "amount produced." Omit if the interface does not surface a count.

Purely human-driven work can use `(human)`. Mixed human + agent sessions can combine them, e.g. `(human + grok-4.3, high)`.

Keep the parenthetical compact. Examples:
- `(grok-4.3, high, ~18k out)`
- `(gpt-4o, high, ~22k out)`
- `(claude-opus-4-7, extended thinking, ~30k out)`
- `(grok-4.3, low)`

Newest entry goes on top. If the session did multiple distinct pieces of work, use multiple `###` subsections under one `##` date header.
-->

## 2026-06-16

### Traced the data pipeline; created parent WORKSPACE_OVERVIEW.md; sharpened project_overview data section (claude-opus-4-8, default thinking)

- Compared the parent-workspace prototype scripts (`../preprocessing.py`, `../make_augmented_data.py`) to the repo's `utils/preprocessing.py` + `write_training_data.py`: same core algorithm; the repo versions added `has_labels` (inference on unlabeled `.mat`), a reusable `write_data()` API, and the `train_val_split.txt` manifest. Minor off-by-one in the REM-window upper bound (`len-seq_len+1` parent vs `len-seq_len` repo; both stay in-bounds). The parent scripts are the originals; the repo versions are the maintained descendants.
- Mapped the external data flow and disambiguated the sibling dirs: source `.mat` files (68) at `../20231006_new_data_for_testing_the_model/Mie_newdata/preprocessed_data/`; **current** training tensors at `../sdreamer_data/n_seq_64/fold_1/` (train `[9269,64,2,1,512]` float32, read by `run_train_sdreamer.py`); `../sdreamer_output_data_augmented_10/seq/...` is an earlier ×10-augmented set (float64, no manifest, superseded); `../sdreamer_output_data*/` and `../sdreamer_datan_seq_64/` are empty scaffolds; `../sdreamer_input_data/` holds older per-recording `.npy`.
- Created `../WORKSPACE_OVERVIEW.md` (parent dir, **not** version-controlled) with the full workspace/data map. Updated `project_overview.md`: concrete `.mat` path + tensor shapes (the old pointer was a stale `C:/Users/...` placeholder), a prototype-lineage note, and a sharper sibling-dir note.
- Verification: tensor shapes/dtypes read via `np.load(..., mmap_mode='r')`; dir contents confirmed with `find`. Docs-only; no pipeline code changed, no training run.

## 2026-06-15

### Committed + pushed the treaty adoption; parked Mie edit moved to stash; origin switched to SSH (claude-opus-4-8, default thinking)

- Committed the treaty docs (`AGENTS.md`, `project_overview.md`, `next_steps.md`, `work_log.md` + `work_log_archive/`, `.copier-answers.yml`) plus the treaty-adopted SVG badge in `README.md` as `bfb8b15` on `dev`, then fast-forwarded `main` to it. `origin/dev` and `origin/main` are both at `bfb8b15`.
- The earlier "uncommitted working-tree" Mie data-prep edit to `write_training_data.py` is now in `git stash` (`stash@{0}`): local `main` was 9 commits behind `origin/main` and the upstream file had been reformatted, so the stale edit could not ride onto `dev` cleanly. See `next_steps.md` — reapply by hand, do not `git stash pop` blindly.
- Switched the `origin` remote from HTTPS to SSH (`git@github.com:yzhaoinuw/sdreamer_flow.git`) and deleted the expired `../PAT.txt`; HTTPS/PAT push auth was failing, the SSH key works from compute nodes.
- Verification: pushes reported `867e8e5..bfb8b15 dev -> dev` and `b882cab..bfb8b15 dev -> main`; `git branch -vv` shows local `main`/`dev` synced to origin at `bfb8b15`.

### Filled in the agent-collab treaty docs after a full repo/architecture read (claude-opus-4-8, default thinking)

- Read the active pipeline end-to-end: `run_train_sdreamer.py` (config/entrypoint) → `exp/exp_moe2.py` (`Exp_MoE`) → `models/seq/n2nSeqNewMoE2.py` (`Model`) → `layers/` (transformer/attention/patch/head), plus `write_training_data.py` + `utils/preprocessing.py` (data prep) and `data_provider/` (`Seq_Loader`).
- Filled in `AGENTS.md` placeholders: runtime env (`conda activate sleep_scoring_dash3.0`), common tasks (data prep + train commands, no test suite), pre-flight checklist, and a Project-Specific Reminders section capturing the non-obvious gotchas.
- Rewrote `project_overview.md` in full: what sDREAMER is, the 4-stage active runtime path, a hyperparameter table, repo structure map, active-vs-legacy map, data expectations, mental-model reading order, and open questions.
- Updated `next_steps.md` with the real in-flight thread (data prep on the new Mie dataset) and parked the CRF and NE threads as background.
- Key facts recorded for future agents: only `SeqNewMoE2`/`SeqHMoE` are live (rest of `models/`+`exp/` is legacy); `scale=0.0` disables the self-distillation; `weight=[1,1,1]` silently becomes unweighted CE (`len < 3` guard); classes are 0=Wake/1=NREM/2=REM; the `*\r` files in the repo root are stray PostScript junk; training data/checkpoints live in sibling `../sdreamer_*` dirs, not in-repo; `README.md`'s file-structure block is stale.
- Verification: documentation-only session; no code changed. Findings cross-checked directly against the source files cited above (no training run performed).
