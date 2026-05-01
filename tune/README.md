# tune/ — Optuna-driven hyperparameter tuning for CATSO/PATSO

Phase 1 scope: Optuna TPE study driving a single C++ "eval" binary per env. Single
fidelity (one trial budget per Optuna trial), single objective (minimize
`cvar_regret` averaged over `num_seeds`). Hyperband / multi-fidelity is Phase 3.

## One-time setup

```bash
source .venv/bin/activate
pip install -r tune/requirements.txt

# Build the eval binary for whichever env you're tuning
make mcts-eval-bettinggame
```

## Run a study

```bash
# Phase 1: only bettinggame is wired
python3 tune/tune.py --env bettinggame --algo CATSO --n-trials 50

# Resume a study (Optuna SQLite storage is persistent)
python3 tune/tune.py --env bettinggame --algo CATSO --n-trials 50 --study-name catso_bg_v1

# Inspect best config
python3 tune/tune.py --env bettinggame --algo CATSO --best-only --study-name catso_bg_v1
```

Outputs land in `tune/results/<env>/<algo>/`:
- `study.db` — Optuna SQLite storage (resumable)
- `best.json` — winning hyperparameters + metrics
- `trials.csv` — flat dump of every trial for plotting

## How it works

1. Python `tune/tune.py` creates an Optuna study with TPE sampler.
2. Each trial samples hyperparams from `tune/search_spaces.yaml`.
3. Python invokes the C++ `mcts-eval-<env>` binary as a subprocess with the
   sampled hyperparams as CLI args.
4. The binary runs MCTS with `num_seeds` independent seeds, computes per-seed
   metrics, and emits one JSON line to stdout.
5. Python parses the JSON, returns `cvar_regret` (mean over seeds) as the
   objective.

Static eval settings (trial budget, seeds, eval rollouts, horizon, τ) live in
`tune/env_configs.yaml`. Search spaces live in `tune/search_spaces.yaml`.

## Phase 2 / 3 roadmap

- Phase 2: wire `eval_autonomous_vehicle` and `eval_guarded_maze`. The Python
  driver already supports them via env_configs — just uncomment.
- Phase 3: multi-fidelity. Replace the static `trial_budget` with a fidelity
  ladder (e.g., 5k / 25k / 100k) and add `optuna.pruners.HyperbandPruner`.
  The eval binary already takes `--trial-budget` so no C++ changes needed.
