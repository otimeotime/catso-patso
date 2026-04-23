#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>
#include <string>
#include <utility>

namespace mcts::exp {
    /**
     * Sailing gridworld (6x6 in the paper figure) with 8-connected movement and stochastic wind.
     *
     * State is (x,y,wind_dir) encoded as Int3TupleState.
     * - wind_dir is an integer in [0,7] for the 8 compass directions.
     * - Wind changes stochastically after each transition.
     * - Actions are the 8 compass moves.
     * - Directly sailing into the wind is not allowed (excluded from valid actions).
     *
     * Reward is dense and negative: reward = -cost(action, wind_dir).
     */
    class SailingEnv : public mcts::MctsEnv {
        private:
            int grid_size;
            int init_wind_dir;

            // Wind transition model: from d, new d is d + {-1,0,+1} mod 8 with given probs.
            double wind_stay_prob;
            double wind_turn_prob; // probability for each of left/right

            double base_cost;
            double angle_cost_scale;
            double goal_reward;  // reward for reaching the goal

            bool in_bounds(int x, int y) const;
            std::pair<int, int> move_delta_from_action(const std::string& action) const;
            int action_dir_index(const std::string& action) const;
            std::string dir_index_to_action(int dir) const;

            bool is_into_wind(int action_dir, int wind_dir) const;
            double compute_cost(int action_dir, int wind_dir) const;

        public:
            SailingEnv(
                int grid_size = 6,
                int init_wind_dir = 0,
                double wind_stay_prob = 0.5,
                double wind_turn_prob = 0.25,
                double base_cost = 1.0,
                double angle_cost_scale = 1.0,
                double goal_reward = 10.0);

            virtual ~SailingEnv() = default;

            // Typed API
            std::shared_ptr<const mcts::Int3TupleState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::Int3TupleState> state) const;
            std::shared_ptr<mcts::StringActionVector> get_valid_actions(std::shared_ptr<const mcts::Int3TupleState> state) const;
            std::shared_ptr<mcts::Int3TupleStateDistr> get_transition_distribution(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::StringAction> action) const;
            std::shared_ptr<const mcts::Int3TupleState> sample_transition_distribution(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::StringAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::StringAction> action,
                std::shared_ptr<const mcts::Int3TupleState> observation = nullptr) const;

            // Interface API
            virtual std::shared_ptr<const mcts::State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const mcts::State> state) const;
            virtual std::shared_ptr<mcts::ActionVector> get_valid_actions_itfc(std::shared_ptr<const mcts::State> state) const;
            virtual std::shared_ptr<mcts::StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action) const;
            virtual std::shared_ptr<const mcts::State> sample_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                mcts::RandManager& rand_manager) const;
            virtual double get_reward_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                std::shared_ptr<const mcts::Observation> observation = nullptr) const;

            int get_grid_size() const { return grid_size; }
    };
}