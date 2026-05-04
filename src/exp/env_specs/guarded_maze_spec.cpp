#include "exp/experiment_spec.h"

#include "env/guarded_maze_env.h"
#include "exp/env_specs/generic_discrete_env_spec_helpers.h"
#include "exp/oracles/guarded_maze_cvar_oracle.h"

#include <array>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>

namespace mcts::exp {
    namespace {
        struct GuardedMazeEnvConfig {
            mcts::exp::GuardedMazeEnv::MazeLayout maze_layout =
                mcts::exp::GuardedMazeEnv::default_maze_layout();
            std::array<int, 2> start = {6, 1};
            std::array<int, 2> goal = {5, 6};
            std::array<int, 2> guard = {6, 5};
            int max_steps = 100;
            int deduction_limit = 100;
            double reward_normalisation = 256.0;
        };

        std::string format_cell(const std::array<int, 2>& cell) {
            std::ostringstream os;
            os << "(" << cell[0] << "," << cell[1] << ")";
            return os.str();
        }
    }

    ExperimentSpec make_guarded_maze_spec() {
        ExperimentSpec spec;
        // Edit this block to change the maze layout and environment dynamics.
        const GuardedMazeEnvConfig env_config{};

        constexpr int kPlanningHorizon = 10;
        constexpr double kCvarTau = 0.25;
        constexpr int kEvalRollouts = 200;
        constexpr int kRuns = 50;
        constexpr int kThreads = 8;
        constexpr int kBaseSeed = 4242;

        spec.env_name = "guarded_maze";
        spec.display_name = "GuardedMaze";
        spec.output_prefix = "results_guarded_maze";
        spec.tune_output_csv = "tune_guarded_maze.csv";
        spec.horizon = kPlanningHorizon;
        spec.eval_rollouts = kEvalRollouts;
        spec.runs = kRuns;
        spec.threads = kThreads;
        spec.base_seed = kBaseSeed;
        spec.cvar_tau = kCvarTau;
        spec.discount_gamma = 1.0;
        spec.trial_counts = {100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};

        mcts::exp::env_specs::configure_uct_family_run_config(
            spec.run_candidate_config,
            0.1,
            mcts::UctManagerArgs::USE_AUTO_BIAS,
            true,
            1.0);
        mcts::exp::env_specs::configure_distributional_run_config(
            spec.run_candidate_config,
            100,
            1.0,
            128,
            4.0,
            1.0);

        mcts::exp::env_specs::configure_uct_family_tuning_grid(
            spec.tuning_grid_config,
            {mcts::UctManagerArgs::USE_AUTO_BIAS, 2.0, 5.0, 10.0},
            {0.05, 0.1, 0.2, 0.5},
            {1.0, 2.0, 4.0, 8.0});
        mcts::exp::env_specs::configure_distributional_tuning_grid(
            spec.tuning_grid_config,
            {25, 51, 100},
            {0.5, 1.0, 2.0, 4.0},
            {32, 64, 128},
            {0.5, 1.0, 2.0, 4.0},
            {1.0, 2.0, 4.0, 8.0},
            {0.05, 0.1, 0.2, 0.25});

        spec.make_env = [env_config]() -> std::shared_ptr<mcts::MctsEnv> {
            return std::make_shared<mcts::exp::GuardedMazeEnv>(
                env_config.maze_layout,
                env_config.start,
                env_config.goal,
                env_config.guard,
                env_config.max_steps,
                env_config.deduction_limit,
                env_config.reward_normalisation);
        };

        spec.print_env_info = [env_config](std::ostream& os) {
            os << "[exp] GuardedMaze"
               << ", grid=" << env_config.maze_layout.size()
               << "x" << env_config.maze_layout.front().size()
               << ", start=" << format_cell(env_config.start)
               << ", goal=" << format_cell(env_config.goal)
               << ", guard=" << format_cell(env_config.guard)
               << ", env_max_steps=" << env_config.max_steps
               << ", planner_horizon=" << kPlanningHorizon
               << ", cvar_tau=" << kCvarTau
               << "\n";
        };

        auto oracle_cache =
            std::make_shared<std::map<
                std::pair<double, double>,
                std::shared_ptr<mcts::exp::oracles::GuardedMazeCvarOracle>>>();
        spec.evaluate_root_metrics =
            [oracle_cache](
                std::shared_ptr<const mcts::MctsEnv> generic_env,
                std::shared_ptr<const mcts::MctsDNode> root,
                double eval_tau,
                double discount_gamma)
            {
                const auto env = std::static_pointer_cast<const mcts::exp::GuardedMazeEnv>(generic_env);
                auto& oracle = (*oracle_cache)[{eval_tau, discount_gamma}];
                if (!oracle) {
                    oracle = std::make_shared<mcts::exp::oracles::GuardedMazeCvarOracle>(
                        env,
                        eval_tau,
                        discount_gamma);
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
