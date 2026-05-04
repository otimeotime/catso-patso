# readme_exp.md — Running CATSO/PATSO experiments

End-to-end guide for the experiment workflow on this repo: building the binaries, running an environment, tuning hyperparameters, and plotting the results.

---

## 1. Prerequisites

- C++20 compiler (`g++` by default, also works with `clang++`).
- GNU `make`.
- For plotting only: Python 3 with `matplotlib`. `plot.py` is a single-file script (no notebook required); `plot.ipynb` is kept around as the equivalent notebook for interactive use.
- (Optional) LaTeX in `PATH` — both `plot.py` and `plot.ipynb` auto-detect it and use TeX-style fonts if present, otherwise fall back to STIX.

No Python venv is required for running the C++ experiments; only for the plotter. A `.venv/` is provided — activate with `source .venv/bin/activate` before running `plot.py`.

---

## 2. Repo layout cheat-sheet

| Path | Purpose |
|---|---|
| `include/algorithms/catso/` | CATSO + PATSO headers (managers, decision/chance nodes) |
| `src/catso/` | CATSO + PATSO implementation |
| `include/env/` , `src/env/` | All environment implementations (`betting_game_env`, `guarded_maze_env`, …) |
| `src/exp/run_<env>.cpp` | One main() per environment — runs UCT + CATSO + PATSO and writes CSVs |
| `src/exp/run_experiment.cpp` / `src/exp/tune_experiment.cpp` | Generic experiment entrypoints, driven by the env registry |
| `src/exp/tune_<env>.cpp` | Older per-env tuners kept for reference; `mcts-tune --env <name>` is the primary path now |
| `include/exp/discrete_runner_common.h` | Shared template the boilerplate runners delegate to |
| `plot.py` | CLI plotter — reads a `results_<env>_summary.csv` and emits 5 PDF plots into `plots/` |
| `plot.ipynb` | Notebook equivalent of `plot.py` (for interactive use) |
| `Makefile` | Build rules — `make <target>` to build each binary |

---

## 3. Environments wired to CATSO/PATSO

All 12 envs below are available through the generic `mcts-run --env <name>` and `mcts-tune --env <name>` entrypoints. The dedicated `mcts-run-<env>` binaries remain available for comparison with the older workflow.

| Env | `--env` value | τ (CVaR α) | runs |
|---|---|---|---|
| Autonomous Vehicle | `autonomous_vehicle` | 0.10 | 3 |
| Betting Game | `bettinggame` | 0.20 | 3 |
| Gambler Jackpot | `gambler_jackpot` | 0.10 | 3 |
| Guarded Maze | `guarded_maze` | 0.10 | 3 |
| Laser Tag Safe Grid | `laser_tag_safe_grid` | 0.05 | 3 |
| Overflow Queue | `overflow_queue` | 0.05 | 3 |
| Risky Ladder | `risky_ladder` | 0.05 | 3 |
| Risky Shortcut Gridworld | `risky_shortcut_gridworld` | 0.05 | 1 |
| River Swim Stochastic | `river_swim_stochastic` | 0.10 | 3 |
| Thin Ice Frozen Lake Plus | `thin_ice_frozen_lake_plus` | 0.05 | 3 |
| Two-level Risky Treasure | `two_level_risky_treasure` | 0.05 | 10 |
| Two-Path SSP Deceptive | `two_path_ssp_deceptive` | 0.05 | 3 |

Every env in the table above is now wired into the generic tuner via `mcts-tune --env <name>`.

---

## 4. Quick-start workflow (3 commands)

The fastest path from clean checkout to plots:

```bash
# 1) Build the generic runner (~1 min, builds the whole tree once)
make mcts-run

# 2) Run one environment (this is the long step — typically hours for full sweeps)
./mcts-run --env bettinggame

# 3) Plot — one command, writes 5 PDFs into plots/
source .venv/bin/activate
python3 plot.py results_bettinggame_summary.csv
```

Outputs land in the repo root:
- `results_bettinggame.csv` — per-run, per-trial-budget rows
- `results_bettinggame_summary.csv` — averaged over `runs` runs (this is what plotting uses)
- `plots/bettinggame*.pdf` — five PDF files after the notebook runs

---

## 5. Detailed workflow

### 5.1. Build

```bash
make mcts-run
```

Use `./mcts-run --env <env_name>` for the generic path. The dedicated `mcts-run-<env>` binaries still work, but they are no longer required for the 12 experiment envs listed in §3. The first build compiles all common objects + every env (~1–2 min); subsequent builds only re-link.

To rebuild after touching CATSO/PATSO code:

```bash
make clean && make mcts-run
```

`make clean` deletes everything under `bin/` and removes all `mcts-run-*` / `mcts-tune-*` binaries.

### 5.2. Run an experiment

```bash
./mcts-run --env <env_name>
```

The runner prints:
- environment metadata (horizon, win_prob, reward bounds, …)
- one line per (algorithm × run) showing progress

Behaviour:
- Each run plans from the initial state with a growing simulation budget (typically `1k, 2k, …, 100k`), and at every budget snapshot evaluates the tree by Monte-Carlo rollouts.
- Per-run seeds are deterministic (`base_seed + 1000·run + hash(algo)`), so reruns are reproducible.
- Multithreading comes from the registered env spec (`threads`), typically 4 or 8 depending on the environment.

CLI flags:
- `mcts-run` accepts `--env <env_name>`.
- `mcts-tune` accepts `--env <env_name>`.
- The dedicated `mcts-run-<env>` binaries still read their config from the `constexpr` values in `src/exp/run_<env>.cpp`.

To change settings in the generic path (for example horizon, runs, or trial budgets), edit the registered env spec and rebuild:

```cpp
// e.g. src/exp/env_specs/additional_env_specs.cpp
spec.horizon = 50;
spec.eval_rollouts = 50;
spec.runs = 3;
spec.threads = 8;
spec.trial_counts = {1000, 2000, 10000, 20000};
```

### 5.3. Tune hyperparameters

```bash
make mcts-tune
./mcts-tune --env bettinggame
```

The tuner sweeps:
- CATSO: `n_atoms ∈ {25, 51, 100}`, `optimism ∈ {2, 4, 8}`, `p ∈ {1, 2, 4}`, `τ ∈ {0.05, 0.1, 0.2, 0.25}`
- PATSO: `max_particles ∈ {32, 64, 128}`, same `optimism × p × τ` grid

Output: `tune_<env>.csv` (one row per (algorithm, config, τ, run, trial)). Pick the best config by sorting on `cvar_regret` or `optimal_action_prob` for your target τ:

```bash
# example: sort CATSO @ 100k trials, τ=0.25, by mean cvar_regret
awk -F, 'NR==1 || ($2=="CATSO" && $4==0.25 && $6==100000)' tune_bettinggame.csv \
  | sort -t, -k9,9g | head
```

Once you've picked a config, copy it into the corresponding env spec or dedicated runner if you want to make it the default.

### 5.4. Plot

The recommended path is `plot.py` (CLI, one command). The notebook `plot.ipynb` is kept around for interactive use and produces identical output.

```bash
source .venv/bin/activate     # activate the provided virtual env (matplotlib lives there)
python3 plot.py results_<env>_summary.csv
```

Common flags:

```bash
# Custom output directory and title:
python3 plot.py results_guarded_maze_summary.csv --outdir plots_v2 --title "Guarded Maze v2"

# Plot a subset of algorithms:
python3 plot.py results_bettinggame_summary.csv --algos CATSO PATSO

# Custom colors (must match --algos length):
python3 plot.py results_bettinggame_summary.csv --algos UCT CATSO PATSO --colors black tab:red tab:purple

# Help:
python3 plot.py --help
```

It produces five plots into `plots/` named `<env_id>*.pdf`:
- `<env_id>.pdf` — Monte-Carlo mean return ± stddev band
- `<env_id>_mc_cvar.pdf` — Monte-Carlo CVaR estimate ± stddev band
- `<env_id>_cvar_regret.pdf` — CVaR regret vs. trial budget
- `<env_id>_optimal_action_prob.pdf` — P(root recommendation matches CVaR-optimal action)
- `<env_id>_catastrophic_count.pdf` — count of evaluation rollouts hitting a "catastrophic" state

#### `mc_cvar_stddev` column (handled automatically by `plot.py`)

Only `bettinggame` and `autonomous_vehicle` summary CSVs include `mc_cvar_stddev`. The other 10 runners write a summary without it. `plot.py` treats this column as optional — it skips the stddev band on the `mc_cvar` plot when missing and prints a one-line note. (The notebook is stricter: edit cell 4's `required` set if you use it on a boilerplate env.)

If you want the band on every env, the cleaner fix is to copy the `cvar_stddev` plumbing from `run_autonomous_vehicle.cpp` (look for `compute_empirical_lower_tail_stats`) into `include/exp/discrete_runner_common.h`. Then every env gets the column for free.

#### Plotting multiple envs in one go

`plot.py` takes one CSV per invocation. To batch-plot, loop in your shell:

```bash
source .venv/bin/activate
for f in results_*_summary.csv; do python3 plot.py "$f"; done
```

The env name is read from each CSV's `env` field, so output PDFs don't collide.

---

## 6. Hyperparameter knobs at a glance

These are the four fields the candidate builders set on `CatsoManagerArgs` / `PatsoManagerArgs`:

| Field | What it does | Paper sweep range | Where it's read |
|---|---|---|---|
| `n_atoms` (CATSO only) | Number of fixed atoms `z_0…z_{N-1}` in the categorical posterior | {10, 20, …, 100} | `catso_chance_node.cpp` (Cramér projection) |
| `max_particles` (PATSO only) | Cap K on adaptive particle list — exceeding triggers closest-pair merge | {50, 100, 200, 400} | `patso_chance_node.cpp::maybe_merge_particles` |
| `optimism_constant` (C) | Coefficient on polynomial exploration bonus `C·T_s^{1/4}/√T_{s,a}` | {4, 8, 16} | `catso_decision_node.cpp::compute_optimism_bonus` |
| `power_mean_exponent` (p) | Power for the V-node power mean over child Q-means | {1, 2, 4} | `catso_decision_node.cpp::compute_power_mean_value` |
| `cvar_tau` (α) | Tail fraction for the CVaR objective at the root | depends on env risk profile | `catso_chance_node.cpp::compute_lower_tail_cvar` |

The first four are independent per algorithm; `cvar_tau` is shared and should match the τ you actually care about evaluating at.

Note: `optimism_constant = 1.0` in 10 of the 12 boilerplate runners — that's outside the paper's swept range and is the most likely cause of underexploration on those envs. Re-tune before drawing strong conclusions from those plots.

---

## 7. Where everything lands

After a full run from a clean repo:

```
catso-patso/
├── mcts-run                             # generic runner
├── mcts-tune                            # generic tuner
├── results_<env>.csv                    # per-trial, per-run rows
├── results_<env>_summary.csv            # averaged over `runs` — feeds plot.ipynb
├── tune_<env>.csv                       # per-config rows from tuner
├── plots/                               # PDF plots from notebook
│   └── <env_id>{,_mc_cvar,_cvar_regret,_optimal_action_prob,_catastrophic_count}.pdf
└── bin/                                 # build artifacts
```

`*.csv`, `plots/`, `bin/`, `*.o`, `*.d`, and the `mcts-run-*` / `mcts-tune-*` binaries are all gitignored, so you can mix experiment output freely with source edits.

---

## 8. Running multiple experiments efficiently

A few practical tips:

- **In parallel across envs:** the runners themselves multithread and usually use 4 or 8 threads; running two `./mcts-run --env ...` jobs in parallel can easily oversubscribe cores. On an 8-core box, run them sequentially. On 16+ cores, you can run two at once.
- **Long runs in the background:** use `nohup ./mcts-run --env bettinggame > log_betting.txt 2>&1 &`, then `tail -f log_betting.txt` to monitor. The runners log a "<algo> run X/N done" line per run completion, so progress is easy to track.
- **Smoke-test first:** before committing to a full sweep, shrink the registered env spec's `runs` and `trial_counts`, rebuild, and run for a minute. Confirms everything links + writes valid CSV before you spend hours.
- **Reproducibility:** `kBaseSeed` (default 4242) is the only randomness source. Same seed + same code + same hardware threading → same results. If you compare runs across code changes, keep the seed fixed.

---

## 9. Common gotchas

- **`results_<env>.csv` overwrite:** the runner truncates the file every time. If you want to keep a previous result, rename or move it before running again.
- **Build-time vs run-time config:** there are no env vars or config files for hyperparameters. Everything is compiled in. After editing an env spec or dedicated runner, you must rebuild.
- **Killing a running experiment:** `Ctrl+C` is safe; the partial CSV is whatever was flushed before the kill. Buffering means the most recent few lines may be missing.
- **`mc_cvar_stddev`:** `plot.py` handles its absence gracefully. The notebook does not — see §5.4.
- **`mc_eval` falls back to random off-tree:** if the post-tree evaluation rollout reaches a state not present in the searched tree (because the budget was small or the env is highly stochastic), `EvalPolicy` switches to uniform random. With `eval_rollouts = 200`, low budgets can therefore look noisier than the underlying planner is — increase the budget, not the eval count, if curves look jagged at low trial counts.

---

## 10. Adding a new environment with CATSO/PATSO

Step by step:

1. **Implement the env** — subclass `MctsEnv` in `include/env/<name>_env.h` and `src/env/<name>_env.cpp`. Mirror an existing one (`betting_game_env` is a good small reference).
2. **Add the env spec** — register a `make_<name>_spec()` factory under `src/exp/env_specs/`, set `make_env`, `trial_counts`, default hyperparameters, and any catastrophe/root-metric callbacks.
3. **Register it** — add the factory to [register_all_envs.cpp](/home/sonlh/trung/catso-patso/src/exp/register_all_envs.cpp:17) so `mcts-run --env <name>` and `mcts-tune --env <name>` can discover it.
4. **Build + smoke-test** as in §8 with a reduced spec (`runs=1`, tiny `trial_counts`).
5. **Plot** — `python3 plot.py results_<name>_summary.csv` after activating the venv.
