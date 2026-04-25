#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>
#include <unordered_set>
#include <utility>

namespace mcts::exp {
    class ThinIceFrozenLakePlusEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 4;
            static constexpr int up_action = 0;
            static constexpr int right_action = 1;
            static constexpr int down_action = 2;
            static constexpr int left_action = 3;

            static constexpr int active_status = 0;
            static constexpr int goal_status = 1;
            static constexpr int broken_status = 2;
            static constexpr int truncated_status = 3;

        private:
            int N;
            double slip_prob;
            double break_prob;
            double step_cost;
            double goal_reward;
            double break_penalty;
            int max_steps;
            std::unordered_set<int> thin_cells;

            int encode_cell(int row, int col) const;
            bool is_thin(int row, int col) const;
            std::pair<int, int> clip(int row, int col) const;
            void decode_state(
                std::shared_ptr<const mcts::Int4TupleState> state,
                int& row,
                int& col,
                int& steps,
                int& status) const;
            std::shared_ptr<const mcts::Int4TupleState> make_state(int row, int col, int steps, int status) const;

        public:
            ThinIceFrozenLakePlusEnv(
                int N = 8,
                double slip_prob = 0.15,
                double thin_frac = 0.15,
                double break_prob = 0.10,
                double step_cost = -1.0,
                double goal_reward = 40.0,
                double break_penalty = -120.0,
                int max_steps = -1,
                int seed = 1);
            virtual ~ThinIceFrozenLakePlusEnv() = default;

            std::shared_ptr<const mcts::Int4TupleState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::Int4TupleState> state) const;
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
