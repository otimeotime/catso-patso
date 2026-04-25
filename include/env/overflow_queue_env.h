#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>

namespace mcts::exp {
    class OverflowQueueEnv : public mcts::MctsEnv {
        public:
            static constexpr int active_status = 0;
            static constexpr int overflow_terminal_status = 1;
            static constexpr int truncated_status = 2;

        private:
            int K;
            int S_max;
            double burst_prob;
            int normal_high;
            int burst_low;
            int burst_high;
            double hold_cost;
            double service_cost;
            double overflow_penalty;
            bool terminal_on_overflow;
            int max_steps;

            void decode_state(
                std::shared_ptr<const mcts::Int4TupleState> state,
                int& queue_len,
                int& steps,
                int& status,
                int& last_arrivals) const;
            std::shared_ptr<const mcts::Int4TupleState> make_state(
                int queue_len,
                int steps,
                int status,
                int last_arrivals) const;

        public:
            OverflowQueueEnv(
                int K = 20,
                int S_max = 5,
                double burst_prob = 0.05,
                int normal_high = 2,
                int burst_low = 6,
                int burst_high = 12,
                double hold_cost = -0.5,
                double service_cost = -0.1,
                double overflow_penalty = -200.0,
                bool terminal_on_overflow = true,
                int max_steps = 200);
            virtual ~OverflowQueueEnv() = default;

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
