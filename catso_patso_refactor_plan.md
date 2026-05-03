# Refactor Plan for `catso-patso` Experiment Runner and Tuning Workflow

## Goal

The current repository has one full run or tune source file per environment, for example:

```text
run_bettinggame.cpp
run_autonomous_vehicle.cpp
run_guarded_maze.cpp
run_risky_ladder.cpp
...
```

This creates a lot of duplicated work when adding a new environment. The goal of this refactor is to replace that pattern with:

```text
one generic run program
one generic tune program
a shared environment registry
small environment-specific spec files
shared algorithm factories
shared evaluation utilities
optional environment-specific oracle plugins
```

After the refactor, instead of running separate binaries such as:

```bash
./mcts-run-bettinggame
./mcts-run-autonomous-vehicle
./mcts-run-guarded-maze
```

you should be able to run:

```bash
./mcts-run --env bettinggame
./mcts-run --env autonomous_vehicle
./mcts-run --env guarded_maze
```

and similarly:

```bash
./mcts-tune --env bettinggame
./mcts-tune --env autonomous_vehicle
```

---

## Target Architecture

The final structure should look like this:

```text
src/exp/
  run_experiment.cpp              # one generic run main
  tune_experiment.cpp             # one generic tune main

  experiment_spec.h               # describes one environment experiment
  experiment_runner.h
  experiment_runner.cpp           # generic run logic

  tuning_runner.h
  tuning_runner.cpp               # generic tuning logic

  evaluation_utils.h
  evaluation_utils.cpp            # MC evaluation, CVaR, stats

  algorithm_factory.h
  algorithm_factory.cpp           # UCT/CATSO/PATSO builders

  env_registry.h
  env_registry.cpp                # maps env name -> env spec
  register_all_envs.cpp

  env_specs/
    bettinggame_spec.cpp
    autonomous_vehicle_spec.cpp
    guarded_maze_spec.cpp
    ...

  oracles/
    bettinggame_cvar_oracle.h
    bettinggame_cvar_oracle.cpp
    autonomous_vehicle_cvar_oracle.h
    autonomous_vehicle_cvar_oracle.cpp
    ...
```

The key idea is:

```text
Generic runner owns the experiment loop.
Environment spec owns environment construction and environment-specific parameters.
Algorithm factory owns UCT/CATSO/PATSO construction.
Evaluation utilities own MC evaluation and CVaR computation.
Oracle plugins own exact optimal-action and regret computations when available.
```

---

# Phase 0 — Freeze Current Behavior

Before refactoring, save a baseline.

Build and run the current binaries:

```bash
make mcts-run-bettinggame
./mcts-run-bettinggame

make mcts-run-autonomous-vehicle
./mcts-run-autonomous-vehicle
```

Save the generated files:

```text
results_bettinggame.csv
results_bettinggame_summary.csv
results_autonomous_vehicle.csv
results_autonomous_vehicle_summary.csv
```

These files are useful for checking that the new generic runner produces the same CSV format and roughly the same behavior.

Do **not** do a big rewrite before saving baseline outputs.

---

# Phase 1 — Extract Common Evaluation Utilities

Files like `run_bettinggame.cpp`, `run_autonomous_vehicle.cpp`, and `tune_bettinggame.cpp` repeat CVaR and evaluation logic.

Create:

```text
src/exp/evaluation_utils.h
src/exp/evaluation_utils.cpp
```

Move these shared structs and functions there:

```cpp
struct EvalStats {
    double mean = 0.0;
    double stddev = 0.0;
    double cvar = 0.0;
    double cvar_stddev = 0.0;
    int catastrophic_count = 0;
};

struct TailStats {
    double mean = 0.0;
    double stddev = 0.0;
};

using ReturnDistribution = std::map<double, double>;

double compute_lower_tail_cvar(
    const ReturnDistribution& distribution,
    double tau
);

TailStats compute_empirical_lower_tail_stats(
    std::vector<double> samples,
    double tau
);

EvalStats evaluate_tree_generic(
    std::shared_ptr<const mcts::MctsEnv> env,
    std::shared_ptr<const mcts::MctsDNode> root,
    int horizon,
    int rollouts,
    int threads,
    int seed,
    double cvar_tau
);
```

The generic `evaluate_tree_generic` should only depend on the `MctsEnv` interface:

```cpp
shared_ptr<const mcts::State> state = env->get_initial_state_itfc();
env->is_sink_state_itfc(state);
env->sample_context_itfc(state);
env->sample_transition_distribution_itfc(state, action, rng);
env->sample_observation_distribution_itfc(action, next_state, rng);
env->get_reward_itfc(state, action, observation);
env->is_catastrophic_state_itfc(next_state);
```

## Definition of Done

After this phase, these files should no longer define their own copies of CVaR and MC evaluation helpers:

```text
run_bettinggame.cpp
run_autonomous_vehicle.cpp
tune_bettinggame.cpp
```

They should include:

```cpp
#include "exp/evaluation_utils.h"
```

---

# Phase 2 — Extract the Algorithm Factory

The current run files manually build UCT, CATSO, and PATSO candidates. Move that code into one shared factory.

Create:

```text
src/exp/algorithm_factory.h
src/exp/algorithm_factory.cpp
```

Suggested header:

```cpp
#pragma once

#include "mcts.h"
#include "mcts_env.h"
#include "mcts_decision_node.h"
#include "mcts_manager.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mcts::exp {

struct Candidate {
    std::string algo;
    std::string config;
    double eval_tau = 0.0;

    std::function<std::pair<
        std::shared_ptr<mcts::MctsManager>,
        std::shared_ptr<mcts::MctsDNode>
    >(
        std::shared_ptr<mcts::MctsEnv> env,
        std::shared_ptr<const mcts::State> init_state,
        int max_depth,
        int seed
    )> make;
};

std::vector<Candidate> build_default_run_candidates(double cvar_tau);

std::vector<Candidate> build_default_tuning_candidates(double default_eval_tau);

}
```

Implementation should contain your current UCT/CATSO/PATSO construction.

You should have two builders:

```cpp
build_default_run_candidates(cvar_tau);
build_default_tuning_candidates(default_eval_tau);
```

The run version uses your final tuned configurations.

The tuning version expands the parameter grid.

Example tuning grid:

```text
UCT:
  bias in {auto, 2.0, 5.0, 10.0}
  epsilon in {0.05, 0.1, 0.2, 0.5}

CATSO:
  n_atoms in {25, 51, 100}
  optimism in {2.0, 4.0, 8.0}
  p in {1.0, 2.0, 4.0}
  tau in {0.05, 0.1, 0.2, 0.25}

PATSO:
  max_particles in {32, 64, 128}
  optimism in {2.0, 4.0, 8.0}
  p in {1.0, 2.0, 4.0}
  tau in {0.05, 0.1, 0.2, 0.25}
```

## Definition of Done

Old environment-specific files should no longer have:

```cpp
static vector<Candidate> build_candidates(...)
```

They should call:

```cpp
auto candidates = mcts::exp::build_default_run_candidates(cvar_tau);
```

or:

```cpp
auto candidates = mcts::exp::build_default_tuning_candidates(cvar_tau);
```

---

# Phase 3 — Define `ExperimentSpec`

Create a shared object that describes one environment experiment.

Create:

```text
src/exp/experiment_spec.h
```

Suggested content:

```cpp
#pragma once

#include "mcts_env.h"
#include "mcts_decision_node.h"

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace mcts::exp {

struct RootMetrics {
    double cvar_regret = std::numeric_limits<double>::quiet_NaN();
    double optimal_action_hit = std::numeric_limits<double>::quiet_NaN();
};

struct ExperimentSpec {
    std::string env_name;
    std::string output_prefix;

    int horizon = 0;
    int eval_rollouts = 200;
    int runs = 1;
    int threads = 1;
    int base_seed = 4242;
    double cvar_tau = 0.25;

    std::vector<int> trial_counts;

    std::function<std::shared_ptr<mcts::MctsEnv>()> make_env;

    std::function<RootMetrics(
        std::shared_ptr<const mcts::MctsEnv> env,
        std::shared_ptr<const mcts::MctsDNode> root
    )> evaluate_root_metrics;

    std::function<void(std::ostream&)> print_env_info;
};

}
```

Important design choice:

```cpp
evaluate_root_metrics
```

should be optional. Some environments have exact oracle/regret computation. Some do not.

If no oracle is available, the generic runner should output:

```cpp
RootMetrics metrics;
metrics.cvar_regret = NaN;
metrics.optimal_action_hit = NaN;
```

This prevents the generic runner from depending on any specific environment type.

---

# Phase 4 — Create Environment Spec Files

Each environment gets a small spec file.

Create:

```text
src/exp/env_specs/bettinggame_spec.cpp
src/exp/env_specs/autonomous_vehicle_spec.cpp
```

## BettingGame Spec

```cpp
#include "exp/experiment_spec.h"
#include "env/betting_game_env.h"

namespace mcts::exp {

ExperimentSpec make_bettinggame_spec() {
    ExperimentSpec spec;

    const double win_prob = 0.8;
    const int max_sequence_length = 6;
    const double max_state_value = BettingGameEnv::default_max_state_value;
    const double initial_state = BettingGameEnv::default_initial_state;
    const double reward_normalisation = BettingGameEnv::default_reward_normalisation;

    spec.env_name = "bettinggame";
    spec.output_prefix = "results_bettinggame";
    spec.horizon = max_sequence_length;
    spec.eval_rollouts = 200;
    spec.runs = 100;
    spec.threads = 8;
    spec.base_seed = 4242;
    spec.cvar_tau = 0.25;

    for (int i = 1; i <= 100; ++i) {
        spec.trial_counts.push_back(i * 1000);
    }

    spec.make_env = [=]() {
        return std::make_shared<BettingGameEnv>(
            win_prob,
            max_sequence_length,
            max_state_value,
            initial_state,
            reward_normalisation
        );
    };

    spec.print_env_info = [=](std::ostream& os) {
        os << "[exp] BettingGame"
           << ", horizon=" << max_sequence_length
           << ", win_prob=" << win_prob
           << ", max_sequence_length=" << max_sequence_length
           << ", initial_state=" << initial_state
           << ", max_state_value=" << max_state_value
           << ", cvar_tau=" << spec.cvar_tau
           << "\n";
    };

    return spec;
}

}
```

## AutonomousVehicle Spec

```cpp
#include "exp/experiment_spec.h"
#include "env/autonomous_vehicle_env.h"

namespace mcts::exp {

ExperimentSpec make_autonomous_vehicle_spec() {
    ExperimentSpec spec;

    const auto horizontal_edges = AutonomousVehicleEnv::default_horizontal_edges();
    const auto vertical_edges = AutonomousVehicleEnv::default_vertical_edges();
    const auto edge_rewards = AutonomousVehicleEnv::default_edge_rewards();
    const auto edge_probs = AutonomousVehicleEnv::default_edge_probs();

    const int max_steps = 8;
    const double reward_normalisation = AutonomousVehicleEnv::default_reward_normalisation;

    spec.env_name = "autonomous_vehicle";
    spec.output_prefix = "results_autonomous_vehicle";
    spec.horizon = max_steps;
    spec.eval_rollouts = 200;
    spec.runs = 3;
    spec.threads = 8;
    spec.base_seed = 4242;
    spec.cvar_tau = 0.05;

    for (int i = 1; i <= 30; ++i) {
        spec.trial_counts.push_back(i * 1000);
    }

    spec.make_env = [=]() {
        return std::make_shared<AutonomousVehicleEnv>(
            horizontal_edges,
            vertical_edges,
            edge_rewards,
            edge_probs,
            max_steps,
            reward_normalisation
        );
    };

    spec.print_env_info = [=](std::ostream& os) {
        os << "[exp] AutonomousVehicle"
           << ", max_steps=" << max_steps
           << ", edge_probs=[" << edge_probs[0] << "," << edge_probs[1] << "]"
           << ", cvar_tau=" << spec.cvar_tau
           << "\n";
    };

    return spec;
}

}
```

At this stage, do **not** move exact oracle logic yet. First make the generic runner work with MC metrics only.

---

# Phase 5 — Create Environment Registry

Create:

```text
src/exp/env_registry.h
src/exp/env_registry.cpp
src/exp/register_all_envs.cpp
```

## Header

```cpp
#pragma once

#include "exp/experiment_spec.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mcts::exp {

using EnvSpecFactory = std::function<ExperimentSpec()>;

class EnvRegistry {
public:
    static EnvRegistry& instance();

    void register_env(const std::string& name, EnvSpecFactory factory);
    ExperimentSpec make(const std::string& name) const;
    std::vector<std::string> names() const;

private:
    std::unordered_map<std::string, EnvSpecFactory> factories;
};

void register_all_envs();

}
```

## Implementation

```cpp
#include "exp/env_registry.h"

#include <stdexcept>

namespace mcts::exp {

EnvRegistry& EnvRegistry::instance() {
    static EnvRegistry registry;
    return registry;
}

void EnvRegistry::register_env(
    const std::string& name,
    EnvSpecFactory factory
) {
    factories[name] = std::move(factory);
}

ExperimentSpec EnvRegistry::make(const std::string& name) const {
    auto it = factories.find(name);

    if (it == factories.end()) {
        throw std::runtime_error("Unknown environment: " + name);
    }

    return it->second();
}

std::vector<std::string> EnvRegistry::names() const {
    std::vector<std::string> result;

    for (const auto& [name, _] : factories) {
        result.push_back(name);
    }

    return result;
}

}
```

## Register All Environment Specs

```cpp
#include "exp/env_registry.h"

namespace mcts::exp {

ExperimentSpec make_bettinggame_spec();
ExperimentSpec make_autonomous_vehicle_spec();

void register_all_envs() {
    EnvRegistry::instance().register_env("bettinggame", make_bettinggame_spec);
    EnvRegistry::instance().register_env("autonomous_vehicle", make_autonomous_vehicle_spec);
}

}
```

Later, adding GuardedMaze only requires:

```cpp
ExperimentSpec make_guarded_maze_spec();
```

and:

```cpp
EnvRegistry::instance().register_env("guarded_maze", make_guarded_maze_spec);
```

---

# Phase 6 — Implement Generic Experiment Runner

Create:

```text
src/exp/experiment_runner.h
src/exp/experiment_runner.cpp
```

## Header

```cpp
#pragma once

#include "exp/experiment_spec.h"
#include "exp/algorithm_factory.h"

#include <vector>

namespace mcts::exp {

int run_experiment(
    const ExperimentSpec& spec,
    const std::vector<Candidate>& candidates
);

}
```

## Implementation Outline

```cpp
#include "exp/experiment_runner.h"
#include "exp/evaluation_utils.h"
#include "exp/manager_config_printer.h"

#include "mcts.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>

namespace mcts::exp {

struct SummaryAccumulator {
    int count = 0;
    int regret_count = 0;

    double sum_mc_mean = 0.0;
    double sum_mc_stddev = 0.0;
    double sum_mc_cvar = 0.0;
    double sum_mc_cvar_stddev = 0.0;
    double sum_cvar_regret = 0.0;
    double sum_optimal_action_hit = 0.0;
    double sum_catastrophic_count = 0.0;
};

int run_experiment(
    const ExperimentSpec& spec,
    const std::vector<Candidate>& candidates
) {
    auto env = spec.make_env();
    auto init_state = env->get_initial_state_itfc();

    std::map<std::pair<std::string, int>, SummaryAccumulator> summary_by_algo_and_trial;

    const std::string raw_csv = spec.output_prefix + ".csv";
    const std::string summary_csv = spec.output_prefix + "_summary.csv";

    std::ofstream out(raw_csv, std::ios::out | std::ios::trunc);
    out << std::setprecision(17);

    out << "env,algorithm,run,trial,"
        << "mc_mean,mc_stddev,mc_cvar,mc_cvar_stddev,"
        << "cvar_regret,optimal_action_hit,catastrophic_count\n";

    if (spec.print_env_info) {
        spec.print_env_info(std::cout);
    } else {
        std::cout << "[exp] " << spec.env_name << "\n";
    }

    std::cout << "  Algorithms: " << candidates.size()
              << ", Runs: " << spec.runs
              << ", Trial counts: " << spec.trial_counts.size()
              << "\n";

    for (const auto& cand : candidates) {
        bool printed_config = false;

        for (int run = 0; run < spec.runs; ++run) {
            int seed =
                spec.base_seed
                + 1000 * run
                + static_cast<int>(std::hash<std::string>{}(cand.algo) % 997);

            auto [mgr, root] = cand.make(
                env,
                init_state,
                spec.horizon,
                seed
            );

            if (!printed_config) {
                const std::string runtime_config =
                    describe_manager_config(
                        std::static_pointer_cast<const mcts::MctsManager>(mgr)
                    );

                std::cout << "\nRunning " << cand.algo;

                if (!runtime_config.empty()) {
                    std::cout << " [" << runtime_config << "]";
                } else if (!cand.config.empty()) {
                    std::cout << " [" << cand.config << "]";
                }

                std::cout << "...\n";
                printed_config = true;
            }

            auto pool = std::make_unique<mcts::MctsPool>(
                mgr,
                root,
                spec.threads
            );

            int trials_so_far = 0;

            for (int target_trials : spec.trial_counts) {
                const int trials_to_add = target_trials - trials_so_far;

                if (trials_to_add > 0) {
                    pool->run_trials(
                        trials_to_add,
                        std::numeric_limits<double>::max(),
                        true
                    );

                    trials_so_far = target_trials;
                }

                auto stats = evaluate_tree_generic(
                    env,
                    root,
                    spec.horizon,
                    spec.eval_rollouts,
                    spec.threads,
                    seed + 777,
                    spec.cvar_tau
                );

                RootMetrics metrics;

                if (spec.evaluate_root_metrics) {
                    metrics = spec.evaluate_root_metrics(env, root);
                }

                out << spec.env_name << "," << cand.algo << "," << run
                    << "," << target_trials
                    << "," << stats.mean
                    << "," << stats.stddev
                    << "," << stats.cvar
                    << "," << stats.cvar_stddev
                    << "," << metrics.cvar_regret
                    << "," << metrics.optimal_action_hit
                    << "," << stats.catastrophic_count
                    << "\n";

                auto& summary =
                    summary_by_algo_and_trial[{cand.algo, target_trials}];

                summary.count++;
                summary.sum_mc_mean += stats.mean;
                summary.sum_mc_stddev += stats.stddev;
                summary.sum_mc_cvar += stats.cvar;
                summary.sum_mc_cvar_stddev += stats.cvar_stddev;
                summary.sum_catastrophic_count += stats.catastrophic_count;

                if (!std::isnan(metrics.cvar_regret)) {
                    summary.regret_count++;
                    summary.sum_cvar_regret += metrics.cvar_regret;
                }

                if (!std::isnan(metrics.optimal_action_hit)) {
                    summary.sum_optimal_action_hit += metrics.optimal_action_hit;
                }
            }

            std::cout << "  " << cand.algo
                      << " run " << run + 1
                      << "/" << spec.runs
                      << " done\n";
        }
    }

    std::ofstream summary_out(summary_csv, std::ios::out | std::ios::trunc);
    summary_out << std::setprecision(17);

    summary_out << "env,algorithm,trial,"
                << "mc_mean,mc_stddev,mc_cvar,mc_cvar_stddev,"
                << "cvar_regret,optimal_action_prob,catastrophic_count\n";

    for (const auto& [algo_trial, summary] : summary_by_algo_and_trial) {
        const auto& algo = algo_trial.first;
        const int trial = algo_trial.second;

        const double mean_cvar_regret =
            summary.regret_count > 0
            ? summary.sum_cvar_regret / static_cast<double>(summary.regret_count)
            : std::numeric_limits<double>::quiet_NaN();

        summary_out << spec.env_name << "," << algo << "," << trial
                    << "," << summary.sum_mc_mean / summary.count
                    << "," << summary.sum_mc_stddev / summary.count
                    << "," << summary.sum_mc_cvar / summary.count
                    << "," << summary.sum_mc_cvar_stddev / summary.count
                    << "," << mean_cvar_regret
                    << "," << summary.sum_optimal_action_hit / summary.count
                    << "," << summary.sum_catastrophic_count / summary.count
                    << "\n";
    }

    std::cout << "\nResults written to "
              << raw_csv << " and " << summary_csv << "\n";

    return 0;
}

}
```

---

# Phase 7 — Create One Generic Run Binary

Create:

```text
src/exp/run_experiment.cpp
```

```cpp
#include "exp/env_registry.h"
#include "exp/algorithm_factory.h"
#include "exp/experiment_runner.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string env_name;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--env" && i + 1 < argc) {
            env_name = argv[++i];
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " --env <env_name>\n";
            return 0;
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    mcts::exp::register_all_envs();

    if (env_name.empty()) {
        std::cerr << "Missing --env\n";
        std::cerr << "Available environments:\n";

        for (const auto& name : mcts::exp::EnvRegistry::instance().names()) {
            std::cerr << "  " << name << "\n";
        }

        return 1;
    }

    try {
        auto spec = mcts::exp::EnvRegistry::instance().make(env_name);
        auto candidates =
            mcts::exp::build_default_run_candidates(spec.cvar_tau);

        return mcts::exp::run_experiment(spec, candidates);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
```

Then you can run:

```bash
./mcts-run --env bettinggame
./mcts-run --env autonomous_vehicle
```

---

# Phase 8 — Refactor Tuning Separately

Do not mix tuning and running too early. After generic run works, refactor tuning.

Create:

```text
src/exp/tuning_runner.h
src/exp/tuning_runner.cpp
src/exp/tune_experiment.cpp
```

The generic tune loop should reuse:

```cpp
ExperimentSpec
Candidate
evaluate_tree_generic
```

The main difference from run mode is:

```cpp
auto candidates = build_default_tuning_candidates(spec.cvar_tau);
```

and output:

```text
tune_<env>.csv
```

For each candidate:

```text
for candidate
  for run
    create env
    create manager/root
    run total_trials
    evaluate
    write CSV
aggregate best config by:
  algorithm
  eval_tau
  config
```

Add tuning fields to `ExperimentSpec`:

```cpp
int tune_total_trials = 10000;
int tune_runs = 2;
int progress_batch_trials = 10000;
```

Alternative: create a separate `TuningSpec`, but initially it is simpler to keep these in `ExperimentSpec`.

---

# Phase 9 — Move Exact CVaR Oracles into Optional Plugins

Do not put exact oracle logic into the generic runner.

Create:

```text
src/exp/oracles/
  bettinggame_cvar_oracle.h
  bettinggame_cvar_oracle.cpp
  autonomous_vehicle_cvar_oracle.h
  autonomous_vehicle_cvar_oracle.cpp
```

Then in `bettinggame_spec.cpp`:

```cpp
#include "exp/oracles/bettinggame_cvar_oracle.h"
```

Attach the oracle like this:

```cpp
spec.evaluate_root_metrics =
    [=](
        std::shared_ptr<const mcts::MctsEnv> generic_env,
        std::shared_ptr<const mcts::MctsDNode> root
    ) {
        auto env =
            std::static_pointer_cast<const BettingGameEnv>(generic_env);

        BettingGameCvarOracle oracle(env, spec.cvar_tau);
        const auto& root_solution =
            oracle.solve_state(env->get_initial_state());

        return evaluate_bettinggame_root_metrics(
            env,
            root,
            root_solution,
            0.0
        );
    };
```

Do the same for `AutonomousVehicle`.

This keeps the exact regret computation separate from the generic experiment loop.

---

# Phase 10 — Simplify the Makefile

Currently the Makefile manually declares many run/tune source lists and targets.

After refactor, add generic experiment sources:

```makefile
EXP_COMMON_SOURCES = \
    src/exp/algorithm_factory.cpp \
    src/exp/evaluation_utils.cpp \
    src/exp/experiment_runner.cpp \
    src/exp/tuning_runner.cpp \
    src/exp/env_registry.cpp \
    src/exp/register_all_envs.cpp \
    $(wildcard src/exp/env_specs/*.cpp) \
    $(wildcard src/exp/oracles/*.cpp)

RUN_EXPERIMENT_SOURCES = src/exp/run_experiment.cpp
TUNE_EXPERIMENT_SOURCES = src/exp/tune_experiment.cpp

EXP_COMMON_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(EXP_COMMON_SOURCES))
RUN_EXPERIMENT_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_EXPERIMENT_SOURCES))
TUNE_EXPERIMENT_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_EXPERIMENT_SOURCES))

TARGET_MCTS_RUN_GENERIC = mcts-run
TARGET_MCTS_TUNE_GENERIC = mcts-tune

$(TARGET_MCTS_RUN_GENERIC): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(EXP_COMMON_OBJECTS) $(RUN_EXPERIMENT_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_MCTS_TUNE_GENERIC): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(EXP_COMMON_OBJECTS) $(TUNE_EXPERIMENT_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)
```

You can keep old targets temporarily as compatibility aliases:

```makefile
mcts-run-bettinggame: mcts-run
	@echo "Use ./mcts-run --env bettinggame"
```

Remove old targets after confirming the generic binaries work.

---

# Phase 11 — Add Config Files Later

After the code-based registry works, move parameters into config files.

Do **not** start with JSON immediately unless the repo already has a JSON library. First make the registry/refactor work.

Later structure:

```text
configs/run/bettinggame.json
configs/run/autonomous_vehicle.json
configs/tune/bettinggame.json
configs/tune/autonomous_vehicle.json
```

Example:

```json
{
  "env": "bettinggame",
  "horizon": 6,
  "eval_rollouts": 200,
  "runs": 100,
  "threads": 8,
  "base_seed": 4242,
  "cvar_tau": 0.25,
  "trial_counts": {
    "start": 1000,
    "end": 100000,
    "step": 1000
  },
  "env_params": {
    "win_prob": 0.8,
    "max_sequence_length": 6,
    "initial_state": 16,
    "max_state_value": 256,
    "reward_normalisation": 256
  }
}
```

Then your command becomes:

```bash
./mcts-run --config configs/run/bettinggame.json
```

Again, do this only after the registry version works.

---

# Suggested Implementation Order

Use this exact order:

```text
1. Add evaluation_utils.h/.cpp
2. Modify run_bettinggame.cpp to use evaluation_utils
3. Modify run_autonomous_vehicle.cpp to use evaluation_utils
4. Add algorithm_factory.h/.cpp
5. Modify run_bettinggame.cpp to use algorithm_factory
6. Modify run_autonomous_vehicle.cpp to use algorithm_factory
7. Add experiment_spec.h
8. Add bettinggame_spec.cpp
9. Add env_registry.h/.cpp
10. Add run_experiment.cpp
11. Add experiment_runner.h/.cpp
12. Build ./mcts-run --env bettinggame
13. Build ./mcts-run --env autonomous_vehicle
14. Compare new CSV with old CSV format
15. Move exact oracles into exp/oracles
16. Add generic tune runner
17. Replace old Makefile targets
18. Add config file support
```

This order avoids a dangerous big rewrite.

---

# What Not to Do

Do **not** create this design:

```text
BaseEnvironmentExperiment
BettingGameExperiment : BaseEnvironmentExperiment
AutonomousVehicleExperiment : BaseEnvironmentExperiment
GuardedMazeExperiment : BaseEnvironmentExperiment
...
```

That only moves the duplication into classes. You would still have one large experiment class per environment.

Prefer this design:

```text
One generic runner
Small environment spec
Optional oracle plugin
Shared algorithm factory
Shared evaluation utilities
```

---

# Final Target Workflow

After the refactor, adding a new environment should require only:

```text
1. Implement env/new_env.h and env/new_env.cpp
2. Add src/exp/env_specs/new_env_spec.cpp
3. Register it in register_all_envs.cpp
4. Run:
   ./mcts-run --env new_env
   ./mcts-tune --env new_env
```

You should no longer need to create:

```text
run_new_env.cpp
tune_new_env.cpp
debug_new_env.cpp
Makefile target variables
Makefile object variables
Makefile object rules
Makefile binary rules
Copied CVaR helpers
Copied CSV logic
Copied candidate builders
```

---

# Summary

The refactor should move the repository from this pattern:

```text
One large run/tune file per environment
```

to this pattern:

```text
One generic run/tune pipeline
Environment-specific configuration through small spec files
Shared algorithm construction
Shared evaluation utilities
Optional exact-oracle plugins
```

This is the main design improvement for reducing future environment implementation time.
