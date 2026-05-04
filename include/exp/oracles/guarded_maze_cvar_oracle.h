#pragma once

#include "env/guarded_maze_env.h"
#include "exp/oracles/discrete_cvar_oracle_common.h"

#include <map>
#include <memory>
#include <tuple>

namespace mcts::exp::oracles {
    class GuardedMazeCvarOracle {
        private:
            using StateKey = std::tuple<int, int, int, int>;

            std::shared_ptr<const mcts::exp::GuardedMazeEnv> env;
            double tau;
            double discount_gamma;
            std::map<StateKey, OptimalDistributionSolution> memo;

            static StateKey make_key(std::shared_ptr<const mcts::Int4TupleState> state);

        public:
            GuardedMazeCvarOracle(
                std::shared_ptr<const mcts::exp::GuardedMazeEnv> env,
                double tau,
                double discount_gamma = 1.0);

            const OptimalDistributionSolution& solve_state(
                std::shared_ptr<const mcts::Int4TupleState> state);
    };

    RootMetrics evaluate_guarded_maze_root_metrics(
        std::shared_ptr<const mcts::exp::GuardedMazeEnv> env,
        std::shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution,
        double eval_tau,
        double optimal_action_regret_threshold = 0.0);
}
