#include "exp/experiment_spec.h"

#include "env/guarded_maze_env.h"
#include "exp/oracles/guarded_maze_cvar_oracle.h"

#include <map>
#include <memory>
#include <ostream>

namespace mcts::exp {
    ExperimentSpec make_guarded_maze_spec() {
        ExperimentSpec spec;

        constexpr int kMaxSteps = 10;
        constexpr double kCvarTau = 0.1;
        constexpr int kEvalRollouts = 200;
        constexpr int kRuns = 3;
        constexpr int kThreads = 8;
        constexpr int kBaseSeed = 4242;

        spec.env_name = "guarded_maze";
        spec.display_name = "GuardedMaze";
        spec.output_prefix = "results_guarded_maze";
        spec.tune_output_csv = "tune_guarded_maze.csv";
        spec.horizon = kMaxSteps;
        spec.eval_rollouts = kEvalRollouts;
        spec.runs = kRuns;
        spec.threads = kThreads;
        spec.base_seed = kBaseSeed;
        spec.cvar_tau = kCvarTau;
        spec.trial_counts = {10000, 20000, 30000, 40000, 50000};

        spec.run_candidate_config.catso_n_atoms = 100;
        spec.run_candidate_config.catso_optimism = 1.0;
        spec.run_candidate_config.power_mean_exponent = 1.0;
        spec.run_candidate_config.patso_particles = 128;
        spec.run_candidate_config.patso_optimism = 4.0;

        spec.tuning_grid_config.catso_n_atoms_values = {25, 51, 100};
        spec.tuning_grid_config.catso_optimism_values = {0.5, 1.0, 2.0, 4.0};
        spec.tuning_grid_config.patso_particles_values = {32, 64, 128};
        spec.tuning_grid_config.patso_optimism_values = {0.5, 1.0, 2.0, 4.0};
        spec.tuning_grid_config.tau_values = {0.05, 0.1, 0.2, 0.25};

        spec.make_env = []() -> std::shared_ptr<mcts::MctsEnv> {
            return std::make_shared<mcts::exp::GuardedMazeEnv>();
        };

        spec.print_env_info = [](std::ostream& os) {
            os << "[exp] GuardedMaze"
               << ", grid=8x8"
               << ", start=(6,1)"
               << ", goal=(5,6)"
               << ", guard=(6,5)"
               << ", horizon=" << kMaxSteps
               << ", cvar_tau=" << kCvarTau
               << "\n";
        };

        auto oracle_cache =
            std::make_shared<std::map<double, std::shared_ptr<mcts::exp::oracles::GuardedMazeCvarOracle>>>();
        spec.evaluate_root_metrics =
            [oracle_cache](
                std::shared_ptr<const mcts::MctsEnv> generic_env,
                std::shared_ptr<const mcts::MctsDNode> root,
                double eval_tau)
            {
                const auto env = std::static_pointer_cast<const mcts::exp::GuardedMazeEnv>(generic_env);
                auto& oracle = (*oracle_cache)[eval_tau];
                if (!oracle) {
                    oracle = std::make_shared<mcts::exp::oracles::GuardedMazeCvarOracle>(env, eval_tau);
                }
                const auto& root_solution = oracle->solve_state(env->get_initial_state());
                return mcts::exp::oracles::evaluate_guarded_maze_root_metrics(
                    env,
                    root,
                    root_solution,
                    eval_tau);
            };

        return spec;
    }
}
