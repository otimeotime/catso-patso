#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>
#include <unordered_set>
#include <utility>

namespace mcts::exp {
    class LaserTagSafeGridEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 4;
            static constexpr int up_action = 0;
            static constexpr int right_action = 1;
            static constexpr int down_action = 2;
            static constexpr int left_action = 3;

            static constexpr int active_status = 0;
            static constexpr int goal_status = 1;
            static constexpr int shot_status = 2;
            static constexpr int truncated_status = 3;

        private:
            int N;
            double eps_shot;
            double step_cost;
            double goal_reward;
            double shot_penalty;
            double slip_prob;
            int max_steps;
            std::unordered_set<int> obstacles;

            int encode_cell(int row, int col) const;
            bool is_obstacle(int row, int col) const;
            bool is_free(int row, int col) const;
            std::pair<int, int> step_to(int row, int col, int action) const;
            bool is_exposed(int row, int col) const;
            void decode_state(
                std::shared_ptr<const mcts::Int4TupleState> state,
                int& row,
                int& col,
                int& steps,
                int& status) const;
            std::shared_ptr<const mcts::Int4TupleState> make_state(int row, int col, int steps, int status) const;

        public:
            LaserTagSafeGridEnv(
                int N = 5,
                double obstacles_frac = 0.12,
                double eps_shot = 0.08,
                double step_cost = -1.0,
                double goal_reward = 45.0,
                double shot_penalty = -150.0,
                double slip_prob = 0.10,
                int max_steps = 400,
                int seed = 1);
            virtual ~LaserTagSafeGridEnv() = default;

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
