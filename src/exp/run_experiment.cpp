#include "exp/algorithm_factory.h"
#include "exp/env_registry.h"
#include "exp/experiment_runner.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string env_name;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
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
        const mcts::exp::ExperimentSpec spec = mcts::exp::EnvRegistry::instance().make(env_name);
        const auto candidates =
            mcts::exp::build_default_run_candidates(spec.cvar_tau, spec.run_candidate_config);
        return mcts::exp::run_experiment(spec, candidates);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
