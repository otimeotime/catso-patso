#pragma once

#include "exp/algorithm_factory.h"
#include "mcts_decision_node.h"
#include "mcts_env.h"

#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace mcts::exp {
    struct RootMetrics {
        double cvar_regret = std::numeric_limits<double>::quiet_NaN();
        double optimal_action_hit = std::numeric_limits<double>::quiet_NaN();
    };

    struct ExperimentSpec {
        std::string env_name;
        std::string display_name;
        std::string output_prefix;
        std::string tune_output_csv;

        int horizon = 0;
        int eval_rollouts = 200;
        int runs = 1;
        int threads = 1;
        int base_seed = 4242;
        double cvar_tau = 0.25;

        std::vector<int> trial_counts;

        RunCandidateConfig run_candidate_config;
        TuningGridConfig tuning_grid_config;

        int tune_total_trials = 10000;
        int tune_runs = 2;
        int tune_threads = 0;
        int progress_batch_trials = 10000;

        std::function<std::shared_ptr<mcts::MctsEnv>()> make_env;
        std::function<bool(
            std::shared_ptr<const mcts::State> state,
            std::shared_ptr<const mcts::Action> action,
            std::shared_ptr<const mcts::State> next_state
        )> is_catastrophic_transition;

        std::function<RootMetrics(
            std::shared_ptr<const mcts::MctsEnv> env,
            std::shared_ptr<const mcts::MctsDNode> root,
            double eval_tau
        )> evaluate_root_metrics;

        std::function<void(std::ostream&)> print_env_info;
    };
}
