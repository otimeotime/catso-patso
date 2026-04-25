#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>

namespace mcts::exp {
    class GamblerJackpotEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 3;
            static constexpr int active_status = 0;
            static constexpr int bankrupt_status = 1;
            static constexpr int success_status = 2;
            static constexpr int truncated_status = 3;

        private:
            int capital_target;
            double p_safe;
            double p_risky;
            int jackpot_gain;
            double jackpot_prob;
            double success_bonus;
            double bankruptcy_penalty;
            int max_steps;

            void decode_state(
                std::shared_ptr<const mcts::Int3TupleState> state,
                int& capital,
                int& steps,
                int& status) const;
            std::shared_ptr<const mcts::Int3TupleState> make_state(int capital, int steps, int status) const;

        public:
            GamblerJackpotEnv(
                int capital_target = 50,
                double p_safe = 0.60,
                double p_risky = 0.55,
                int jackpot_gain = 20,
                double jackpot_prob = 0.03,
                double success_bonus = 0.0,
                double bankruptcy_penalty = 0.0,
                int max_steps = 1000);
            virtual ~GamblerJackpotEnv() = default;

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
