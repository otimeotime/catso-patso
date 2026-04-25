#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>

namespace mcts::exp {
    class RiskyLadderEnv : public mcts::MctsEnv {
        public:
            static constexpr int advance_action = 0;
            static constexpr int bail_action = 1;

            static constexpr int active_status = 0;
            static constexpr int bail_status = 1;
            static constexpr int success_status = 2;
            static constexpr int fail_status = 3;
            static constexpr int truncated_status = 4;

        private:
            int H;
            double rung_reward;
            double bonus_reward;
            double fail_prob;
            double fail_penalty;
            int max_steps;

            void decode_state(
                std::shared_ptr<const mcts::Int3TupleState> state,
                int& rung,
                int& steps,
                int& status) const;
            std::shared_ptr<const mcts::Int3TupleState> make_state(int rung, int steps, int status) const;

        public:
            RiskyLadderEnv(
                int H = 8,
                double rung_reward = 5.0,
                double bonus_reward = 60.0,
                double fail_prob = 0.15,
                double fail_penalty = 120.0,
                int max_steps = 50);
            virtual ~RiskyLadderEnv() = default;

            std::shared_ptr<const mcts::Int3TupleState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::Int3TupleState> state) const;
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
