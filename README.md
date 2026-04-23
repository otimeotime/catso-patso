# Monte Carlo Tree Search (MCTS)

C++ implementation and experiment utilities for MCTS-style algorithms.

## Build
From this folder, using a C++20-capable compiler (default: `g++`) and `make`:
- `make mcts-tune-frozenlake`
- `make mcts-tune-sailing`
- `make mcts-tune-taxi`
- `make mcts-tune-stree`
- `make mcts-run-frozenlake`
- `make mcts-run-sailing`
- `make mcts-run-taxi`
- `make mcts-run-stree`

## Outputs
- The Makefile `clean` target removes build artifacts under `mcts/bin/` and produced binaries.
- Plot artifacts typically go under `mcts/plots/` (excluded via `.gitignore`).

## Plotting
- `plot.ipynb` can be used to visualize results (notebook outputs are cleared for submission hygiene).

# catso-patso
