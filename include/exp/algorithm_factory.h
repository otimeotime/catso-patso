#pragma once

#include "algorithms/uct/uct_manager.h"
#include "mcts_decision_node.h"
#include "mcts_env.h"
#include "mcts_manager.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mcts::exp {
    struct RunCandidateConfig {
        double uct_bias = mcts::UctManagerArgs::USE_AUTO_BIAS;
        bool uct_use_auto_bias = true;
        double uct_epsilon = 0.1;

        int catso_n_atoms = 100;
        double catso_optimism = 4.0;
        double power_mean_exponent = 1.0;

        int patso_particles = 100;
        double patso_optimism = 4.0;
    };

    struct TuningGridConfig {
        std::vector<double> uct_bias_values = {
            mcts::UctManagerArgs::USE_AUTO_BIAS,
            2.0,
            5.0,
            10.0
        };
        std::vector<double> uct_epsilon_values = {0.05, 0.1, 0.2, 0.5};

        std::vector<int> catso_n_atoms_values = {25, 51, 100};
        std::vector<double> catso_optimism_values = {2.0, 4.0, 8.0};

        std::vector<int> patso_particles_values = {32, 64, 128};
        std::vector<double> patso_optimism_values = {2.0, 4.0, 8.0};

        // The current repo tuning code fixes p=1.0.
        std::vector<double> power_mean_exponent_values = {1.0, 2.0, 4.0, 8.0};
        std::vector<double> tau_values = {0.05, 0.1, 0.2, 0.25};
    };

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

    std::vector<Candidate> build_default_run_candidates(
        double cvar_tau,
        const RunCandidateConfig& config = {});

    std::vector<Candidate> build_default_tuning_candidates(
        double default_eval_tau,
        const TuningGridConfig& config = {});
}
