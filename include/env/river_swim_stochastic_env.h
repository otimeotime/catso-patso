#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>

namespace mcts::exp {
    class RiverSwimStochasticEnv : public mcts::MctsEnv {
        public:
            static constexpr int left_action = 0;
            static constexpr int right_action = 1;

            static constexpr int active_status = 0;
            static constexpr int truncated_status = 1;

            static constexpr int left_outcome = 0;
            static constexpr int right_success_outcome = 1;
            static constexpr int right_fail_left_outcome = 2;
            static constexpr int right_fail_stay_outcome = 3;

        private:
            int N;
            double p_right_success;
            double p_right_fail_left;
            double left_reward;
            double right_reward;
            int max_steps;

            void decode_state(
                std::shared_ptr<const mcts::Int4TupleState> state,
                int& river_state,
                int& steps,
                int& status,
                int& last_outcome) const;
            std::shared_ptr<const mcts::Int4TupleState> make_state(
                int river_state,
                int steps,
                int status,
                int last_outcome) const;

        public:
            RiverSwimStochasticEnv(
                int N = 7,
                double p_right_success = 0.35,
                double p_right_fail_left = 0.55,
                double left_reward = 0.05,
                double right_reward = 1.0,
                int max_steps = 200);
            virtual ~RiverSwimStochasticEnv() = default;

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
