#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>
#include <unordered_set>
#include <vector>

namespace mcts::exp {
    class RiskyShortcutGridworldEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 4;
            static constexpr int up_action = 0;
            static constexpr int right_action = 1;
            static constexpr int down_action = 2;
            static constexpr int left_action = 3;

            static constexpr int default_grid_size = 10;
            static constexpr double default_slip_prob = 0.10;
            static constexpr double default_wind_prob = 0.20;
            static constexpr double default_step_cost = -1.0;
            static constexpr double default_goal_reward = 50.0;
            static constexpr double default_cliff_penalty = -100.0;

        private:
            int grid_size;
            int max_steps;
            double slip_prob;
            double wind_prob;
            double step_cost;
            double goal_reward;
            double cliff_penalty;
            std::unordered_set<int> windy_cols;

            std::shared_ptr<const mcts::Int3TupleState> make_state(int row, int col, int time) const;
            void decode_state(
                std::shared_ptr<const mcts::Int3TupleState> state,
                int& row,
                int& col,
                int& time) const;
            bool is_action_valid(int row, int col, int action) const;
            std::pair<int, int> clip(int row, int col) const;
            std::pair<int, int> apply_action_deterministic(int row, int col, int action) const;
            bool is_windy_col(int col) const;

        public:
            RiskyShortcutGridworldEnv(
                int grid_size = default_grid_size,
                double slip_prob = default_slip_prob,
                double wind_prob = default_wind_prob,
                std::vector<int> windy_cols = {},
                double step_cost = default_step_cost,
                double goal_reward = default_goal_reward,
                double cliff_penalty = default_cliff_penalty,
                int max_steps = 20);
            virtual ~RiskyShortcutGridworldEnv() = default;

            // Typed API
            std::shared_ptr<const mcts::Int3TupleState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::Int3TupleState> state) const;
            bool is_goal_cell(int row, int col) const;
            bool is_cliff_cell(int row, int col) const;
            bool is_catastrophic_state(std::shared_ptr<const mcts::Int3TupleState> state) const;
            std::shared_ptr<mcts::IntActionVector> get_valid_actions(
                std::shared_ptr<const mcts::Int3TupleState> state) const;
            std::shared_ptr<mcts::Int3TupleStateDistr> get_transition_distribution(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::IntAction> action) const;
            std::shared_ptr<const mcts::Int3TupleState> sample_transition_distribution(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::IntAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::IntAction> action,
                std::shared_ptr<const mcts::Int3TupleState> observation = nullptr) const;

            // Interface API
            virtual std::shared_ptr<const mcts::State> get_initial_state_itfc() const override;
            virtual bool is_sink_state_itfc(std::shared_ptr<const mcts::State> state) const override;
            virtual std::shared_ptr<mcts::ActionVector> get_valid_actions_itfc(
                std::shared_ptr<const mcts::State> state) const override;
            virtual std::shared_ptr<mcts::StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action) const override;
            virtual std::shared_ptr<const mcts::State> sample_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                mcts::RandManager& rand_manager) const override;
            virtual double get_reward_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                std::shared_ptr<const mcts::Observation> observation = nullptr) const override;

            int get_grid_size() const { return grid_size; }
            int get_max_steps() const { return max_steps; }
            double get_slip_prob() const { return slip_prob; }
            double get_wind_prob() const { return wind_prob; }
            std::vector<int> get_windy_cols() const;
            double get_step_cost() const { return step_cost; }
            double get_goal_reward() const { return goal_reward; }
            double get_cliff_penalty() const { return cliff_penalty; }
    };
}
