#pragma once

#include "env/autonomous_vehicle_env.h"
#include "exp/oracles/discrete_cvar_oracle_common.h"

#include <map>
#include <memory>
#include <tuple>

namespace mcts::exp::oracles {
    class AutonomousVehicleCvarOracle {
        private:
            using StateKey = std::tuple<int, int, int>;

            std::shared_ptr<const mcts::exp::AutonomousVehicleEnv> env;
            double tau;
            double discount_gamma;
            std::map<StateKey, OptimalDistributionSolution> memo;

            static StateKey make_key(std::shared_ptr<const mcts::Int3TupleState> state);

        public:
            AutonomousVehicleCvarOracle(
                std::shared_ptr<const mcts::exp::AutonomousVehicleEnv> env,
                double tau,
                double discount_gamma = 1.0);

            const OptimalDistributionSolution& solve_state(
                std::shared_ptr<const mcts::Int3TupleState> state);
    };

    RootMetrics evaluate_autonomous_vehicle_root_metrics(
        std::shared_ptr<const mcts::exp::AutonomousVehicleEnv> env,
        std::shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution,
        double eval_tau,
        double optimal_action_regret_threshold = 0.0);
}
