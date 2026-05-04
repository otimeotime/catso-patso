#include "exp/algorithm_factory.h"
#include "exp/env_registry.h"
#include "exp/tuning_runner.h"

#include <exception>
#include <iostream>
#include <optional>
#include <string>

int main(int argc, char** argv) {
    std::string env_name;
    std::optional<double> gamma_override;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--env" && i + 1 < argc) {
            env_name = argv[++i];
        }
        else if (arg == "--gamma" && i + 1 < argc) {
            try {
                gamma_override = std::stod(argv[++i]);
            }
            catch (const std::exception&) {
                std::cerr << "Invalid --gamma value\n";
                return 1;
            }
            if (*gamma_override < 0.0 || *gamma_override > 1.0) {
                std::cerr << "--gamma must be in [0,1]\n";
                return 1;
            }
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " --env <env_name> [--gamma <0..1>]\n";
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
        mcts::exp::ExperimentSpec spec = mcts::exp::EnvRegistry::instance().make(env_name);
        if (gamma_override.has_value()) {
            spec.discount_gamma = *gamma_override;
        }
        const auto candidates =
            mcts::exp::build_default_tuning_candidates(
                spec.cvar_tau,
                spec.discount_gamma,
                spec.tuning_grid_config);
        return mcts::exp::run_tuning_experiment(spec, candidates);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
