#include "exp/oracles/autonomous_vehicle_cvar_oracle.h"

#include <algorithm>
#include <utility>

namespace mcts::exp::oracles {
    namespace {
        void add_weighted_shifted_distribution(
            ReturnDistribution& out,
            const ReturnDistribution& in,
            double weight,
            double shift)
        {
            for (const auto& [value, probability] : in) {
                out[value + shift] += weight * probability;
            }
        }
    }

    AutonomousVehicleCvarOracle::StateKey AutonomousVehicleCvarOracle::make_key(
        std::shared_ptr<const mcts::Int3TupleState> state)
    {
        return state->state;
    }

    AutonomousVehicleCvarOracle::AutonomousVehicleCvarOracle(
        std::shared_ptr<const mcts::exp::AutonomousVehicleEnv> env,
        double tau)
        : env(std::move(env)),
          tau(tau)
    {
    }

    const OptimalDistributionSolution& AutonomousVehicleCvarOracle::solve_state(
        std::shared_ptr<const mcts::Int3TupleState> state)
    {
        const StateKey key = make_key(state);
        const auto memo_it = memo.find(key);
        if (memo_it != memo.end()) {
            return memo_it->second;
        }

        OptimalDistributionSolution solution;
        if (env->is_sink_state(state)) {
            solution.state_value_distribution[0.0] = 1.0;
            auto [it, inserted] = memo.emplace(key, std::move(solution));
            (void)inserted;
            return it->second;
        }

        auto actions = env->get_valid_actions(state);
        for (const auto& action : *actions) {
            ReturnDistribution action_distribution;
            auto transitions = env->get_transition_distribution(state, action);

            for (const auto& [next_state, probability] : *transitions) {
                const auto& child_solution = solve_state(next_state);
                const double local_reward = env->get_reward(state, action, next_state);
                add_weighted_shifted_distribution(
                    action_distribution,
                    child_solution.state_value_distribution,
                    probability,
                    local_reward);
            }

            const int action_id = action->action;
            const double action_cvar = compute_lower_tail_cvar(action_distribution, tau);
            solution.action_value_distributions.emplace(action_id, action_distribution);
            solution.action_cvars.emplace(action_id, action_cvar);

            if (solution.optimal_actions.empty() || action_cvar > solution.optimal_cvar + kCvarTolerance) {
                solution.optimal_cvar = action_cvar;
                solution.optimal_actions = {action_id};
                solution.canonical_action = action_id;
            }
            else if (std::abs(action_cvar - solution.optimal_cvar) <= kCvarTolerance) {
                solution.optimal_actions.push_back(action_id);
                solution.canonical_action = std::min(solution.canonical_action, action_id);
            }
        }

        if (solution.canonical_action >= 0) {
            solution.state_value_distribution = solution.action_value_distributions.at(solution.canonical_action);
        }

        auto [it, inserted] = memo.emplace(key, std::move(solution));
        (void)inserted;
        return it->second;
    }

    RootMetrics evaluate_autonomous_vehicle_root_metrics(
        std::shared_ptr<const mcts::exp::AutonomousVehicleEnv> env,
        std::shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution,
        double eval_tau,
        double optimal_action_regret_threshold)
    {
        return evaluate_root_action_cvar_metrics(
            std::static_pointer_cast<const mcts::MctsEnv>(env),
            root,
            root_solution,
            eval_tau,
            optimal_action_regret_threshold);
    }
}
