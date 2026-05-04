#include "exp/oracles/bettinggame_cvar_oracle.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>

namespace mcts::exp::oracles {
    namespace {
        void add_weighted_shifted_distribution(
            ReturnDistribution& out,
            const ReturnDistribution& in,
            double weight,
            double shift,
            double scale = 1.0)
        {
            for (const auto& [value, probability] : in) {
                out[scale * value + shift] += weight * probability;
            }
        }
    }

    std::size_t BettingGameCvarOracle::StateKeyHash::operator()(const StateKey& key) const {
        std::size_t seed = std::hash<double>{}(key.bankroll);
        seed ^= std::hash<int>{}(key.time) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }

    BettingGameCvarOracle::StateKey BettingGameCvarOracle::make_key(
        std::shared_ptr<const mcts::exp::BettingGameState> state)
    {
        return {state->bankroll, state->time};
    }

    BettingGameCvarOracle::BettingGameCvarOracle(
        std::shared_ptr<const mcts::exp::BettingGameEnv> env,
        double tau,
        double discount_gamma)
        : env(std::move(env)),
          tau(tau),
          discount_gamma(discount_gamma)
    {
    }

    const OptimalDistributionSolution& BettingGameCvarOracle::solve_state(
        std::shared_ptr<const mcts::exp::BettingGameState> state)
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
                    local_reward,
                    discount_gamma);
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

    RootMetrics evaluate_bettinggame_root_metrics(
        std::shared_ptr<const mcts::exp::BettingGameEnv> env,
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
