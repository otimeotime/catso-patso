#pragma once

#include "env/betting_game_env.h"
#include "exp/oracles/discrete_cvar_oracle_common.h"

#include <memory>
#include <unordered_map>

namespace mcts::exp::oracles {
    class BettingGameCvarOracle {
        private:
            struct StateKey {
                double bankroll = 0.0;
                int time = 0;

                bool operator==(const StateKey& other) const = default;
            };

            struct StateKeyHash {
                std::size_t operator()(const StateKey& key) const;
            };

            std::shared_ptr<const mcts::exp::BettingGameEnv> env;
            double tau;
            std::unordered_map<StateKey, OptimalDistributionSolution, StateKeyHash> memo;

            static StateKey make_key(std::shared_ptr<const mcts::exp::BettingGameState> state);

        public:
            BettingGameCvarOracle(std::shared_ptr<const mcts::exp::BettingGameEnv> env, double tau);

            const OptimalDistributionSolution& solve_state(
                std::shared_ptr<const mcts::exp::BettingGameState> state);
    };

    RootMetrics evaluate_bettinggame_root_metrics(
        std::shared_ptr<const mcts::exp::BettingGameEnv> env,
        std::shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution,
        double eval_tau,
        double optimal_action_regret_threshold = 0.0);
}
