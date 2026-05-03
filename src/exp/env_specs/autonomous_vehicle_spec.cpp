#include "exp/experiment_spec.h"

#include "env/autonomous_vehicle_env.h"
#include "exp/oracles/autonomous_vehicle_cvar_oracle.h"

#include <algorithm>
#include <map>
#include <memory>
#include <ostream>

namespace mcts::exp {
    ExperimentSpec make_autonomous_vehicle_spec() {
        ExperimentSpec spec;

        constexpr int kMaxSteps = 7;
        constexpr double kCvarTau = 0.1;
        constexpr int kEvalRollouts = 50;
        constexpr int kRuns = 3;
        constexpr int kThreads = 8;
        constexpr int kBaseSeed = 4242;

        spec.env_name = "autonomous_vehicle";
        spec.display_name = "AutonomousVehicle";
        spec.output_prefix = "results_autonomous_vehicle";
        spec.tune_output_csv = "tune_autonomous_vehicle.csv";
        spec.horizon = kMaxSteps;
        spec.eval_rollouts = kEvalRollouts;
        spec.runs = kRuns;
        spec.threads = kThreads;
        spec.base_seed = kBaseSeed;
        spec.cvar_tau = kCvarTau;
        spec.trial_counts = {1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 60000};

        spec.run_candidate_config.catso_n_atoms = 100;
        spec.run_candidate_config.catso_optimism = 2.0;
        spec.run_candidate_config.power_mean_exponent = 1.0;
        spec.run_candidate_config.patso_particles = 128;
        spec.run_candidate_config.patso_optimism = 4.0;
        spec.run_candidate_config.uct_epsilon = 0.05;

        spec.tuning_grid_config.catso_n_atoms_values = {25, 51, 100};
        spec.tuning_grid_config.catso_optimism_values = {0.5, 1.0, 2.0, 4.0};
        spec.tuning_grid_config.patso_particles_values = {32, 64, 128};
        spec.tuning_grid_config.patso_optimism_values = {0.5, 1.0, 2.0, 4.0};
        spec.tuning_grid_config.tau_values = {0.05, 0.1, 0.2, 0.25};
        spec.tune_total_trials = 50000;
        spec.tune_runs = 10;
        spec.tune_threads = kThreads;
        spec.progress_batch_trials = 5000;

        const auto horizontal_edges = mcts::exp::AutonomousVehicleEnv::default_horizontal_edges();
        const auto vertical_edges = mcts::exp::AutonomousVehicleEnv::default_vertical_edges();
        const auto edge_rewards = mcts::exp::AutonomousVehicleEnv::default_edge_rewards();
        const auto edge_probs = mcts::exp::AutonomousVehicleEnv::default_edge_probs();
        const auto reward_normalisation = mcts::exp::AutonomousVehicleEnv::default_reward_normalisation;

        spec.make_env = [=]() -> std::shared_ptr<mcts::MctsEnv> {
            return std::make_shared<mcts::exp::AutonomousVehicleEnv>(
                horizontal_edges,
                vertical_edges,
                edge_rewards,
                edge_probs,
                kMaxSteps,
                reward_normalisation);
        };

        spec.is_catastrophic_transition =
            [=](std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                std::shared_ptr<const mcts::State> next_state)
            {
                const auto typed_state = std::static_pointer_cast<const mcts::Int3TupleState>(state);
                const auto typed_action = std::static_pointer_cast<const mcts::IntAction>(action);
                const auto typed_next_state = std::static_pointer_cast<const mcts::Int3TupleState>(next_state);

                const int x = std::get<0>(typed_state->state);
                const int y = std::get<1>(typed_state->state);
                const int observed_outcome =
                    std::get<2>(typed_next_state->state) % (mcts::exp::AutonomousVehicleEnv::num_reward_outcomes + 1);

                if (observed_outcome != mcts::exp::AutonomousVehicleEnv::num_reward_outcomes - 1) {
                    return false;
                }

                int edge_type = -1;
                switch (typed_action->action) {
                    case mcts::exp::AutonomousVehicleEnv::move_positive_y_action:
                        edge_type = vertical_edges[y][x];
                        break;
                    case mcts::exp::AutonomousVehicleEnv::move_positive_x_action:
                        edge_type = horizontal_edges[y][x];
                        break;
                    case mcts::exp::AutonomousVehicleEnv::move_negative_y_action:
                        edge_type = vertical_edges[y - 1][x];
                        break;
                    case mcts::exp::AutonomousVehicleEnv::move_negative_x_action:
                        edge_type = horizontal_edges[y][x - 1];
                        break;
                    default:
                        return false;
                }

                double max_large_cost =
                    edge_rewards.front()[mcts::exp::AutonomousVehicleEnv::num_reward_outcomes - 1];
                for (const auto& rewards : edge_rewards) {
                    max_large_cost = std::max(
                        max_large_cost,
                        rewards[mcts::exp::AutonomousVehicleEnv::num_reward_outcomes - 1]);
                }

                const double edge_large_cost =
                    edge_rewards[edge_type][mcts::exp::AutonomousVehicleEnv::num_reward_outcomes - 1];
                return edge_large_cost + 1e-12 >= max_large_cost;
            };

        spec.print_env_info = [=](std::ostream& os) {
            os << "[exp] AutonomousVehicle"
               << ", horizon=" << kMaxSteps
               << ", edge_probs=[" << edge_probs[0] << "," << edge_probs[1] << "]"
               << ", cvar_tau=" << kCvarTau
               << "\n";
        };

        auto oracle_cache =
            std::make_shared<std::map<double, std::shared_ptr<mcts::exp::oracles::AutonomousVehicleCvarOracle>>>();
        spec.evaluate_root_metrics =
            [oracle_cache](
                std::shared_ptr<const mcts::MctsEnv> generic_env,
                std::shared_ptr<const mcts::MctsDNode> root,
                double eval_tau)
            {
                const auto env = std::static_pointer_cast<const mcts::exp::AutonomousVehicleEnv>(generic_env);
                auto& oracle = (*oracle_cache)[eval_tau];
                if (!oracle) {
                    oracle = std::make_shared<mcts::exp::oracles::AutonomousVehicleCvarOracle>(env, eval_tau);
                }
                const auto& root_solution = oracle->solve_state(env->get_initial_state());
                return mcts::exp::oracles::evaluate_autonomous_vehicle_root_metrics(
                    env,
                    root,
                    root_solution,
                    eval_tau);
            };

        return spec;
    }
}
