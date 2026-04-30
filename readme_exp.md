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
| `src/exp/tune_<env>.cpp` | Grid-search tuner — currently only `tune_bettinggame` is wired to CATSO/PATSO |
| `include/exp/discrete_runner_common.h` | Shared template the boilerplate runners delegate to |
| `plot.py` | CLI plotter — reads a `results_<env>_summary.csv` and emits 5 PDF plots into `plots/` |
| `plot.ipynb` | Notebook equivalent of `plot.py` (for interactive use) |
| `Makefile` | Build rules — `make <target>` to build each binary |

---

## 3. Environments wired to CATSO/PATSO

All 12 envs below build and run UCT + CATSO + PATSO. Three of them have bespoke runners; the other nine go through the shared template in `discrete_runner_common.h`.

| Env | Make target | Runner type | τ (CVaR α) | runs |
|---|---|---|---|---|
| Betting Game | `mcts-run-bettinggame` | bespoke | 0.25 | 100 |
| Risky Shortcut Gridworld | `mcts-run-risky-shortcut-gridworld` | bespoke | 0.05 | 3 |
| Autonomous Vehicle | `mcts-run-autonomous-vehicle` | bespoke | 0.05 | 3 |
| Guarded Maze | `mcts-run-guarded-maze` | template | 0.05 | 10 |
| Gambler Jackpot | `mcts-run-gambler-jackpot` | template | 0.10 | 3 |
| Laser Tag Safe Grid | `mcts-run-laser-tag-safe-grid` | template | 0.05 | 3 |
| Overflow Queue | `mcts-run-overflow-queue` | template | 0.05 | 3 |
| Risky Ladder | `mcts-run-risky-ladder` | template | 0.05 | 3 |
| River Swim Stochastic | `mcts-run-river-swim-stochastic` | template | 0.10 | 3 |
| Thin Ice Frozen Lake Plus | `mcts-run-thin-ice-frozen-lake-plus` | template | 0.05 | 3 |
| Two-level Risky Treasure | `mcts-run-two-level-risky-treasure` | template | 0.05 | 3 |
| Two-Path SSP Deceptive | `mcts-run-two-path-ssp-deceptive` | template | 0.05 | 3 |

Tuners exist only for `mcts-tune-bettinggame` (CATSO/PATSO grid). The rest are intentionally untuned — see §6 for how to add one.

---

## 4. Quick-start workflow (3 commands)

The fastest path from clean checkout to plots:

```bash
# 1) Build the experiment binary (~1 min, builds the whole tree once)
make mcts-run-bettinggame

# 2) Run the experiment (this is the long step — typically hours for full sweeps)
./mcts-run-bettinggame

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
make mcts-run-<env>
```

Replace `<env>` with one of the targets in §3. The first build compiles all common objects + every env (~1–2 min); subsequent builds only re-link.

To rebuild after touching CATSO/PATSO code:

```bash
make clean && make mcts-run-<env>
```

`make clean` deletes everything under `bin/` and removes all `mcts-run-*` / `mcts-tune-*` binaries.

### 5.2. Run an experiment

```bash
./mcts-run-<env>
```

The runner prints:
- environment metadata (horizon, win_prob, reward bounds, …)
- the CVaR oracle's optimal CVaR at the root
- one line per (algorithm × run) showing progress

Behaviour:
- Each run plans from the initial state with a growing simulation budget (typically `1k, 2k, …, 100k`), and at every budget snapshot evaluates the tree by Monte-Carlo rollouts.
- Per-run seeds are deterministic (`base_seed + 1000·run + hash(algo)`), so reruns are reproducible.
- Multithreading: 8 worker threads inside `MctsPool` (hardcoded as `kThreads`).

CLI flags:
- `mcts-run-bettinggame` and `mcts-tune-bettinggame` accept `--optimal-action-regret-threshold <ε>`. Useful for relaxing the "optimal action hit" criterion when there are near-ties at the root (default = 0).
- All other runners ignore command-line arguments — config lives as `constexpr int / double` constants at the top of each `run_<env>.cpp`.

To change settings in those runners (for example horizon, runs, trial budget), edit the constants and rebuild:

```cpp
// e.g. src/exp/run_guarded_maze.cpp
constexpr int kMaxSteps = 500;
constexpr double kCvarTau = 0.05;
constexpr int kEvalRollouts = 200;
constexpr int kRuns = 10;
constexpr int kThreads = 8;
constexpr int kBaseSeed = 4242;
constexpr int kCatsoAtoms = 51;       // CATSO N
constexpr double kOptimism = 1.0;     // C in the polynomial bonus
constexpr double kPowerMeanExponent = 2.0;   // p in V-node power mean
constexpr int kPatsoParticles = 64;   // PATSO K
```

### 5.3. Tune hyperparameters

Currently only Betting Game has a tuner.

```bash
make mcts-tune-bettinggame
./mcts-tune-bettinggame
```

The tuner sweeps:
- CATSO: `n_atoms ∈ {25, 51, 100}`, `optimism ∈ {2, 4, 8}`, `p ∈ {1, 2, 4}`, `τ ∈ {0.05, 0.1, 0.2, 0.25}`
- PATSO: `max_particles ∈ {32, 64, 128}`, same `optimism × p × τ` grid

Output: `tune_bettinggame.csv` (one row per (algorithm, config, τ, run, trial)). Pick the best config by sorting on `cvar_regret` or `optimal_action_prob` for your target τ:

```bash
# example: sort CATSO @ 100k trials, τ=0.25, by mean cvar_regret
awk -F, 'NR==1 || ($2=="CATSO" && $4==0.25 && $6==100000)' tune_bettinggame.csv \
  | sort -t, -k9,9g | head
```

Once you've picked a config, copy it into `src/exp/run_bettinggame.cpp` (`build_candidates`).

#### Adding a tuner for another env

There is no shared tuner template — you'd clone `src/exp/tune_bettinggame.cpp`, swap the env type and oracle, and add Makefile entries (`TUNE_<ENV>_SOURCES`, the link rule, and a target alias). For the boilerplate envs you'd also need a typed oracle solver — `discrete_runner_common.h::CvarOracle` already provides one, so you mostly need to wrap the candidate-builder loop in the same shape as `tune_bettinggame.cpp:415-465`.

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
├── mcts-run-<env>                       # built binary
├── mcts-tune-<env>                      # if applicable
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

- **In parallel across envs:** the runners themselves multithread (8 threads) but bind to one process; running two `./mcts-run-<env>` in parallel just oversubscribes cores. On an 8-core box, run them sequentially. On 16+ cores, you can run two at once.
- **Long runs in the background:** use `nohup ./mcts-run-bettinggame > log_betting.txt 2>&1 &`, then `tail -f log_betting.txt` to monitor. The runners log a "<algo> run X/N done" line per run completion, so progress is easy to track.
- **Smoke-test first:** before committing to a full sweep, edit the runner's `kRuns` to `1` and `trial_counts` to `{1000, 5000}`, rebuild, run for a minute. Confirms everything links + writes valid CSV before you spend hours.
- **Reproducibility:** `kBaseSeed` (default 4242) is the only randomness source. Same seed + same code + same hardware threading → same results. If you compare runs across code changes, keep the seed fixed.

---

## 9. Common gotchas

- **`results_<env>.csv` overwrite:** the runner truncates the file every time. If you want to keep a previous result, rename or move it before running again.
- **Build-time vs run-time config:** there are no env vars or config files for hyperparameters. Everything is compiled in. After editing constants in `run_<env>.cpp`, you must rebuild.
- **Killing a running experiment:** `Ctrl+C` is safe; the partial CSV is whatever was flushed before the kill. Buffering means the most recent few lines may be missing.
- **`mc_cvar_stddev`:** `plot.py` handles its absence gracefully. The notebook does not — see §5.4.
- **`mc_eval` falls back to random off-tree:** if the post-tree evaluation rollout reaches a state not present in the searched tree (because the budget was small or the env is highly stochastic), `EvalPolicy` switches to uniform random. With `eval_rollouts = 200`, low budgets can therefore look noisier than the underlying planner is — increase the budget, not the eval count, if curves look jagged at low trial counts.

---

## 10. Adding a new environment with CATSO/PATSO

Step by step:

1. **Implement the env** — subclass `MctsEnv` in `include/env/<name>_env.h` and `src/env/<name>_env.cpp`. Mirror an existing one (`betting_game_env` is a good small reference).
2. **Add the runner** — clone `src/exp/run_guarded_maze.cpp` (8 lines of config + a delegation to `mcts::exp::runner::run_experiment`) and adjust the env type, state type, catastrophe predicate, and constants.
3. **Hook up Make** — in `Makefile`, add three blocks: `RUN_<NAME>_SOURCES`/`RUN_<NAME>_OBJECTS` near line 80, `TARGET_MCTS_RUN_<NAME>` in the targets list around line 160, and the link rule near line 440.
4. **Build + smoke-test** as in §8 with `kRuns=1`.
5. **Plot** — `python3 plot.py results_<name>_summary.csv` after activating the venv. Works as long as the runner emits the standard column set (the boilerplate template already does).
6. **Tune (optional)** — clone `tune_bettinggame.cpp`, scope the env oracle, sweep on a smaller (k, n) before committing to a full grid.
