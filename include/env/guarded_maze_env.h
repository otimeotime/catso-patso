#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace mcts::exp {
    class GuardedMazeEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 4;
            static constexpr int right_action = 0;
            static constexpr int down_action = 1;
            static constexpr int left_action = 2;
            static constexpr int up_action = 3;

            static constexpr int num_guard_reward_outcomes = 5;
            static constexpr int neutral_guard_reward_outcome = 2;
            using MazeLayout = std::vector<std::string>;

        private:
            MazeLayout maze_layout;
            int num_rows;
            int num_cols;
            std::array<int, 2> start;
            std::array<int, 2> goal;
            std::array<int, 2> guard;
            int max_steps;
            int deduction_limit;
            double reward_normalisation;

            bool is_wall(int row, int col) const;
            bool is_goal_cell(int row, int col) const;
            bool is_guard_cell(int row, int col) const;
            void decode_state(
                std::shared_ptr<const mcts::Int4TupleState> state,
                int& row,
                int& col,
                int& steps,
                int& last_guard_outcome) const;
            std::shared_ptr<const mcts::Int4TupleState> make_state(
                int row,
                int col,
                int steps,
                int last_guard_outcome) const;
            double guard_reward_value(int outcome_index) const;
            double guard_reward_probability(int outcome_index) const;

        public:
            static MazeLayout default_maze_layout();

            GuardedMazeEnv(
                MazeLayout maze_layout = default_maze_layout(),
                std::array<int, 2> start = {6, 1},
                std::array<int, 2> goal = {5, 6},
                std::array<int, 2> guard = {6, 5},
                int max_steps = 100,
                int deduction_limit = 100,
                double reward_normalisation = 256.0);
            virtual ~GuardedMazeEnv() = default;

            std::shared_ptr<const mcts::Int4TupleState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::Int4TupleState> state) const;
            bool is_catastrophic_state(std::shared_ptr<const mcts::Int4TupleState> state) const;
            std::shared_ptr<mcts::IntActionVector> get_valid_actions(
                std::shared_ptr<const mcts::Int4TupleState> state) const;
            std::shared_ptr<mcts::Int4TupleStateDistr> get_transition_distribution(
                std::shared_ptr<const mcts::Int4TupleState> state,
                std::shared_ptr<const mcts::IntAction> action) const;
            std::shared_ptr<const mcts::Int4TupleState> sample_transition_distribution(
                std::shared_ptr<const mcts::Int4TupleState> state,
                std::shared_ptr<const mcts::IntAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const mcts::Int4TupleState> state,
                std::shared_ptr<const mcts::IntAction> action,
                std::shared_ptr<const mcts::Int4TupleState> observation = nullptr) const;

            virtual std::shared_ptr<const mcts::State> get_initial_state_itfc() const override;
            virtual bool is_sink_state_itfc(std::shared_ptr<const mcts::State> state) const override;
            virtual bool is_catastrophic_state_itfc(std::shared_ptr<const mcts::State> state) const override;
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
    };
}
