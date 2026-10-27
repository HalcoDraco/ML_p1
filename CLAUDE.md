# Instructions for Claude

EPFL CS-433 Project 1: predict `_MICHD` (coronary heart disease or heart attack) from the answers to the BRFSS 2015 health survey. Binary classification, scored by F1 on AIcrowd. Deadline: Thursday 2026-10-29, 16:00.

The current status, decisions and next steps are in `PROGRESS.md`, imported here so every new chat starts with them:

@PROGRESS.md

## Where things are

- `PROGRESS.md`: status and next steps. **Update it at the end of every task**: status table, decisions, next steps, and one line in the session log. A new chat must be able to continue from it alone.
- `docs/eda_findings.md`: facts about the data found in the EDA (special codes, feature groups, signal, cleaning and feature rules). Read it before working on cleaning (part 2), transforms or models (part 3).
- `README.md`: how to set up and run each part. `STRUCTURE.md`, section *Implementation notes*: the contract of every module in `p1/`.
- `docs/project1_description.pdf`: the assignment.

## Rules

- Never run git write commands (commit, add, mv, rm, stash, checkout, ...). Use plain `mv`/`rm`; the user commits.
- Repo code uses only the standard library and NumPy, plus matplotlib for plots. No pandas or scikit-learn.
- Every `.py` file must be black-formatted (`uv run black .`) and must not contain the substring "todo" in any casing: the grader scans all `.py` files.
- `implementations.py` must stay self-contained and run on Python 3.9 + NumPy 1.23, the grader's environment.
- Use uv (`uv run ...`, `uv add ...`). Ask before adding tools or dependencies.
- Keep throwaway exploration scripts out of the repo.
