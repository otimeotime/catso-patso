# catso-patso

This archive contains CATSO/PATSO experiment code built on top of an MCTS
codebase. We take the Github repository https://github.com/MWPainter/thts-plus-plus for the backbone of this project.

The intended experiment workflow uses two generic frontends:

- `mcts-run`
- `mcts-tune`

This README explains the project structure and the workflow from a clean archive.

## Project tree

```text
catso-patso/
├── bin/                    stores generated build artifacts and may be absent until you compile the project.
│   └── src/                mirrors the source tree inside the build output directory.
│       ├── algorithms/     contains compiled objects for the search algorithms and related support code.
│       ├── env/            contains compiled objects for environment implementations.
│       └── exp/            contains compiled objects for experiment runners, registry wiring, and evaluation utilities.
├── include/                contains the public headers used across the codebase.
│   ├── algorithms/         groups headers for the search algorithms and their shared interfaces.
│   │   ├── catso/          contains headers for the CATSO and PATSO search variants.
│   │   ├── common/         contains headers for utilities reused across multiple algorithm implementations.
│   │   └── uct/            contains headers for the UCT-style baseline search implementation.
│   ├── distributions/      contains headers for probability distribution helpers used by the experiments.
│   ├── env/                contains headers for the supported benchmark environments.
│   ├── exp/                contains headers for experiment orchestration, evaluation helpers, and registry logic.
│   │   └── oracles/        contains headers for oracle-style helpers used by selected experiments.
│   └── templates/          contains reusable template headers shared across multiple components.
└── src/                    contains the implementation files for the project.
    ├── algorithms/         groups source files for the search algorithms and their shared support code.
    │   ├── catso/          contains the CATSO and PATSO implementation files.
    │   ├── common/         contains shared algorithm utilities and helper implementations.
    │   └── uct/            contains the UCT baseline implementation files.
    ├── distributions/      contains implementation files for probability distribution utilities.
    ├── env/                contains implementation files for the supported environments.
    └── exp/                contains the generic experiment frontends, runners, registry wiring, and evaluation code.
        ├── env_specs/      contains environment-specific experiment specifications exposed through `mcts-run` and `mcts-tune`.
        └── oracles/        contains oracle implementations used by environment-specific experiments.
```

## Start from the zip archive

After extracting the archive:

```bash
unzip catso-patso.zip
cd catso-patso
```

All commands below are meant to be run from this project root.

## Prerequisites

You need:

- a C++20 compiler such as `g++` or `clang++`
- GNU `make`
- Python 3 only if you want to plot CSV results with `plot.py`

## Full experiment workflow

The full experiment setup is based on two generic binaries:

- `mcts-run` for running an environment at fixed configurations
- `mcts-tune` for sweeping hyperparameter grids

### Build

Build the generic runner and tuner with:

```bash
make mcts-run
make mcts-tune
```

For a clean rebuild:

```bash
make clean
make mcts-run
make mcts-tune
```

### Run an experiment

Use the generic runner with:

```bash
./mcts-run --env <env_name>
```

Example:

```bash
./mcts-run --env autonomous_vehicle
```

Supported `--env` values in this archive:

- `autonomous_vehicle`
- `risky_shortcut_gridworld`
- `two_level_risky_treasure`

Expected outputs:

- `results_<env>.csv`
- `results_<env>_summary.csv`

### Tune hyperparameters

Use the generic tuner with:

```bash
./mcts-tune --env <env_name>
```

Example:

```bash
./mcts-tune --env risky_shortcut_gridworld
```

Expected output:

- `tune_<env>.csv`

### Plot results

Once a run has produced a summary CSV:

```bash
python3 plot.py results_<env>_summary.csv
```

Example:

```bash
python3 plot.py results_autonomous_vehicle_summary.csv
```

## Additional build targets

The shipped `Makefile` also provides:

- `make help`
- `make generic`
- `make eval`
- `make mcts-eval-autonomous-vehicle`
- `make mcts-eval-risky-shortcut-gridworld`

## Files typically involved in the generic workflow

The main files involved are:

- `src/exp/run_experiment.cpp`
- `src/exp/tune_experiment.cpp`
- `src/exp/experiment_runner.cpp`
- `src/exp/tuning_runner.cpp`
- `src/exp/env_registry.cpp`
- `src/exp/register_all_envs.cpp`
- `src/exp/env_specs/*.cpp`
- `Makefile`

The `--env` value passed to `mcts-run` or `mcts-tune` must be registered in
`src/exp/register_all_envs.cpp`.

## Output summary

The generic workflow produces:

- `results_<env>.csv`
- `results_<env>_summary.csv`
- `tune_<env>.csv`

The evaluator binaries are separate convenience frontends; they do not replace
`mcts-run` or `mcts-tune`.

## Cleaning build artifacts

To remove build artifacts:

```bash
make clean
```

This removes the `bin/` directory and the binaries produced by the current
`Makefile`.
