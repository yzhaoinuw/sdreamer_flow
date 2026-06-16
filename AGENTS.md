# Guidelines and Tips for Agents

This file is the first thing any agent (Claude, Codex, or other) should read when joining a session on this repo. It defines the runtime, the common tasks, the conventions, and the project-specific reminders. Keep it short and current.

## Startup Rule

At the beginning of a new chat or agent session for this project, read this file first and do not automatically read every markdown file in the repository. Use the [Documentation](#documentation) map below to decide which other files are relevant to the current task.

## Runtime Environment

When running code, tests, or the application for this repository, use:

- conda env `sleep_scoring_dash3.0` (Python + PyTorch/CUDA, einops, timm, scikit-learn, scipy, pytorch_lightning)

Typical startup:

```
conda activate sleep_scoring_dash3.0
```

After activation, use that environment for commands such as:

- data prep: `python write_training_data.py`
- training: `python run_train_sdreamer.py`
- import checks, one-off scripts, etc.

There is no automated test suite in this repo (the Copier `test_command` is empty). "Verification" here means: imports resolve, data prep produces the expected `.npy` files, and a short training run logs sane train/val accuracy. Use targeted import smoke checks instead of a test runner, e.g.:

```
python -c "from exp.exp_moe2 import Exp_MoE; from models.seq import n2nSeqNewMoE2"
```

## Common Tasks

Short recipes for the things you'll usually do in a session. All commands assume the env above is active.

Prepare the training data (`.mat` recordings → train/val `.npy` tensors). Edit the
`data_path` / `save_path` / `on_hold_list` in the `__main__` block first, then:

```
python write_training_data.py
```

Train the model (reads the `.npy` tensors written above; all hyperparameters live in the
`config` dict at the top of the file):

```
python run_train_sdreamer.py
```

There is no separate test suite to run. The closest thing to a fast check is an import smoke
check on the active modules:

```
python -c "from exp.exp_moe2 import Exp_MoE; from models.seq import n2nSeqNewMoE2, n2nSeqHMoE"
```

Pre-flight checklist before committing:

- Code is `black`-formatted (the repo is formatted with Black).
- Active modules still import (run the smoke check above).
- If you touched data prep or the model, confirm a few training steps run and log sane
  train/val accuracy before walking away.
- A new entry has been prepended to `work_log.md` describing what was done (including model +
  version, effort/thinking mode, and token budget if available), intended profiling signal if
  any, and the verification commands that were actually run.

## When To Update Treaty Docs

At the end of any substantive work session, update `work_log.md` unless the user explicitly asks not to document it, says it is off the book, or the exchange was clearly trivial.

A session is substantive when it includes any of:

- file edits
- meaningful validation, debugging, profiling, or artifact inspection
- a technical decision or reversal
- discovered evidence future agents should not have to rediscover
- branch, PR, release, deployment, or environment state changes
- unfinished follow-up that belongs in `next_steps.md`

No work-log entry is usually needed for casual Q&A, explanation-only exchanges with no lasting project state, or tiny one-off commands with no future coordination value.

Log experiments when they produce reusable evidence, a decision, or a warning for future agents, even if the code change is reverted. Skip pure scratch work when it has no future coordination value or the user wants it omitted.

When a session creates or changes future work, update `next_steps.md` in the same pass: add concrete follow-ups, remove completed items, and keep "Currently Hot" accurate.

## Branch Handoff Discipline

Before switching away from an experimental or feature branch, fully resolve the work on that branch. Confirm whether the branch contains all intended changes, whether those changes are committed, and whether the user expects them merged, pushed, or intentionally left parked.

Do not switch to the integration branch (`main`) or start new work on another branch while important experimental-branch changes are only local, unmerged, or unverified. If related work accidentally lands on the integration branch, move that work back onto the experimental branch first and retest the combined behavior there before updating the integration branch.

Useful checks before switching or merging (portable git commands; run in any shell):

```
git status --short --branch
git log --oneline --left-right --cherry-pick main...HEAD
git merge-base --is-ancestor main HEAD
```

## Documentation

Read these documents only as needed. The map below names each file and when it's worth opening.

- `work_log.md` and `work_log_archive/`
  - Use when the task needs recent implementation history, experiment outcomes, or verification breadcrumbs.
  - The live `work_log.md` holds at most the 5 most recent unique calendar dates. Default to reading only the two most recent dated entries.
  - Find date anchors with ripgrep and read only the slice you need:
    `rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md`
  - When older context is needed, open the matching file under `work_log_archive/` by its date-range filename, or grep across both at once:
    `rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md work_log_archive/`
  - When prepending a dated entry, if today's calendar date already has a `## YYYY-MM-DD` header at the top, add a new `###` session subsection under it. Do not start a second `## YYYY-MM-DD` header for the same date.
  - When prepending a new date would push the live log past 5 unique calendar dates, move the oldest 5 dates as a chunk into a new file at `work_log_archive/work_log_<earliest>_to_<latest>.md`. The live file always holds at most 5 unique dates; each archive file always holds exactly 5.

- `next_steps.md`
  - Use when planning or continuing unfinished work from previous sessions.
  - The "Currently Hot" pointer at the top names the active threads — read it first to know what's in flight.
  - Remove items after they are completed. Add new planned follow-ups when they become concrete.

- `project_overview.md`
  - Use when onboarding to the codebase structure or when a task touches an unfamiliar area.
  - The "What Looks Active vs. Legacy" section is the single most important map before editing — many repos accumulate parallel implementations, and this section keeps an agent from editing the wrong file.

- `README.md`
  - Use when changing user-facing setup, packaging, usage, or input-file expectations.

- Treaty badge (in your README)
  - `treaty init` offers (opt-in) a centrally-hosted "Agent Collab Treaty - adopted" badge (Codex blue / Claude amber / Grok dark tri-color SVG, or reliable single-color shields.io). It is a pure visibility signal that links back to this treaty repository. No asset files are added to your project, and the image updates automatically if the design improves later. The badge is fully optional. The shields.io version is the dependable recommendation for GitHub READMEs.

- `CONTRIBUTING.md` (if present)
  - Use when changing collaboration workflow, branch/test expectations, or documentation conventions.

The same anchor-grep pattern works for any structured Markdown doc in the repo — `grep -n '^## ' <file>` for the section map, then a targeted slice read rather than loading the whole file.

## Git Ownership Note

If Git reports a "detected dubious ownership" warning for this repo, mark this repository as safe.

Windows (PowerShell):

```powershell
git config --global --add safe.directory C:/path/to/this/repo
```

macOS / Linux:

```bash
git config --global --add safe.directory "$HOME/path/to/this/repo"
```

This is the preferred fix unless the repository ownership itself needs to be changed at the OS level.

## Pre-commit Note

If your stack uses [pre-commit](https://pre-commit.com) and it cannot write to its default cache location, set a repo-local cache before running it.

Windows (PowerShell):

```powershell
$env:PRE_COMMIT_HOME = "C:\path\to\this\repo\.pre-commit-cache"
python -m pre_commit run --all-files
```

macOS / Linux:

```bash
export PRE_COMMIT_HOME="$PWD/.pre-commit-cache"
python -m pre_commit run --all-files
```

Adjust the formatter / linter invocation for your stack (e.g., `npx lint-staged`, `cargo fmt`, `gofmt -l .`).

## Commit Message Guidelines

Commit messages should use:

- a short title line
- a short body with flat bullet points for additional requested changes when a commit contains multiple user-requested updates

Commit message bullets should describe high-level added or changed behavior, not implementation details.

For feature commits, mention only the user-facing behavior that was added or changed.

Do not mention tests, docs, project memory updates, or behind-the-scenes implementation details in a feature commit message unless that internal work is itself the main purpose of the commit.

## Project-Specific Reminders

Add domain-specific gotchas here. These are the things that aren't obvious from reading the code and that an agent would otherwise have to learn the hard way.

- **One active model path.** Despite the dozens of files under `models/` and `exp/`, the live
  pipeline is only `run_train_sdreamer.py` → `exp/exp_moe2.py` (`Exp_MoE`) →
  `models/seq/n2nSeqNewMoE2.py` (`Model`). In `Exp_MoE._build_model` every model except
  `SeqNewMoE2` and `SeqHMoE` is commented out. Don't edit the other models/scripts expecting it
  to affect training. See `project_overview.md` for the full active-vs-legacy map.

- **Data must be written before training.** `run_train_sdreamer.py` expects the `.npy` tensors
  at `../sdreamer_data/n_seq_64/fold_1/`. The loaders only regenerate from raw `.mat`/`.npy` if
  that directory is *missing* — and they do so from `root_path=""` (cwd), which is almost never
  what you want. Run `write_training_data.py` first; treat the `.npy` cache as the source of truth.

- **`scale=0.0` disables self-distillation.** The "self-distilled" loss
  (`loss1 + (distill_eeg + distill_emg) * scale`) reduces to plain cross-entropy on the mixed
  head at the default `scale=0.0`. Bump `scale` to actually train with the EEG/EMG → mix
  distillation that gives sDREAMER its name.

- **Class-weight quirk.** `_select_criterion` only applies `args.weight` when
  `len(args.weight) < 3`. The default `weight=[1,1,1]` has length 3, so the weight is silently
  set to `None` (unweighted CE). To use real per-class weights you currently need a list of
  length ≠ 3, which is a footgun — read that line before relying on weighting.

- **Three classes: 0 = Wake, 1 = NREM/SWS, 2 = REM.** `write_training_data.py`'s `augment=True`
  upsamples windows around REM (class 2) transitions to counter REM rarity.

- **Stray `*\r` files in the repo root are junk.** Files literally named `argparse\r`, `os\r`,
  `np\r`, `torch\r`, `random\r`, `logging\r` (note the trailing carriage return) are misdirected
  ImageMagick/PostScript dumps from an unrelated session, owned by another user. They are safe to
  ignore or delete; never let them shadow real imports.
