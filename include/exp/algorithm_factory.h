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
    struct UctRunConfig {
        bool enabled = true;
        // double bias = mcts::UctManagerArgs::USE_AUTO_BIAS;
        double bias = 5;
        bool use_auto_bias = true;
        double epsilon = 0.05;
    };

    struct MaxUctRunConfig {
        bool enabled = true;
        double bias = 5;
        bool use_auto_bias = true;
        double epsilon = 0.05;
    };

    struct PowerUctRunConfig {
        bool enabled = true;
        double bias = mcts::UctManagerArgs::USE_AUTO_BIAS;
        bool use_auto_bias = true;
        double epsilon = 0.05;
        double power_mean_exponent = 1.0;
    };

    struct CatsoRunConfig {
        bool enabled = true;
        int n_atoms = 100;
        double optimism = 8.0;
        double power_mean_exponent = 1.0;
    };

    struct PatsoRunConfig {
        bool enabled = true;
        int max_particles = 32;
        double optimism = 8.0;
        double power_mean_exponent = 1.0;
    };

    // Env specs choose the run-time algorithm set by toggling `enabled` in each
    // nested block and then overriding only the hyperparameters they care about.
    struct RunCandidateConfig {
        UctRunConfig uct;
        MaxUctRunConfig max_uct;
        PowerUctRunConfig power_uct;
        CatsoRunConfig catso;
        PatsoRunConfig patso;
    };

    struct UctTuningConfig {
        bool enabled = true;
        std::vector<double> bias_values = {
            mcts::UctManagerArgs::USE_AUTO_BIAS,
            2.0,
            5.0,
            10.0
        };
        std::vector<double> epsilon_values = {0.05, 0.1, 0.2, 0.5};
    };

    struct MaxUctTuningConfig {
        bool enabled = true;
        std::vector<double> bias_values = {
            mcts::UctManagerArgs::USE_AUTO_BIAS,
            2.0,
            5.0,
            10.0
        };
        std::vector<double> epsilon_values = {0.05, 0.1, 0.2, 0.5};
    };

    struct PowerUctTuningConfig {
        bool enabled = true;
        std::vector<double> bias_values = {
            mcts::UctManagerArgs::USE_AUTO_BIAS,
            2.0,
            5.0,
            10.0
        };
        std::vector<double> epsilon_values = {0.05, 0.1, 0.2, 0.5};
        std::vector<double> power_mean_exponent_values = {1.0, 2.0, 4.0, 8.0};
    };

    struct CatsoTuningConfig {
        bool enabled = true;
        std::vector<int> n_atoms_values = {25, 51, 100};
        std::vector<double> optimism_values = {2.0, 4.0, 8.0};
        std::vector<double> power_mean_exponent_values = {1.0};
        std::vector<double> tau_values = {0.05, 0.1, 0.2, 0.25};
    };

    struct PatsoTuningConfig {
        bool enabled = true;
        std::vector<int> max_particles_values = {32, 64, 128};
        std::vector<double> optimism_values = {2.0, 4.0, 8.0};
        std::vector<double> power_mean_exponent_values = {1.0};
        std::vector<double> tau_values = {0.05, 0.1, 0.2, 0.25};
    };

    // Env specs choose the tuning search space the same way: enable the
    // algorithms to tune and customize each per-algorithm grid independently.
    struct TuningGridConfig {
        UctTuningConfig uct;
        MaxUctTuningConfig max_uct;
        PowerUctTuningConfig power_uct;
        CatsoTuningConfig catso;
        PatsoTuningConfig patso;
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
        double discount_gamma,
        const RunCandidateConfig& config = {});

    std::vector<Candidate> build_default_tuning_candidates(
        double default_eval_tau,
        double discount_gamma,
        const TuningGridConfig& config = {});
}
