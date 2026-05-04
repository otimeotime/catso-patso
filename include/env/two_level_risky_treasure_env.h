#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>

namespace mcts::exp {
    class TwoLevelRiskyTreasureEnv : public mcts::MctsEnv {
        public:
            static constexpr int safe_action = 0;
            static constexpr int risky_action = 1;
            static constexpr int abort_action = 0;
            static constexpr int proceed_action = 1;

            static constexpr int root_stage = 0;
            static constexpr int hint_stage = 1;

            static constexpr int no_hint = -1;
            static constexpr int bad_hint = 0;
            static constexpr int good_hint = 1;

            static constexpr int active_status = 0;
            static constexpr int safe_status = 1;
            static constexpr int abort_status = 2;
            static constexpr int success_status = 3;
            static constexpr int fail_status = 4;

        private:
            double safe_reward;
            double success_reward;
            double fail_penalty;
            double prior_success_prob;
            double hint_error;
            double abort_penalty;

            void decode_state(
                std::shared_ptr<const mcts::Int4TupleState> state,
                int& stage,
                int& hint,
                int& steps,
                int& status) const;
            std::shared_ptr<const mcts::Int4TupleState> make_state(int stage, int hint, int steps, int status) const;
            double posterior_success_prob(int hint) const;

        public:
            TwoLevelRiskyTreasureEnv(
                double safe_reward = 2.0,
                double success_reward = 500.0,
                double fail_penalty = 10.0,
                double prior_success_prob = 0.40,
                double hint_error = 0.30,
                double abort_penalty = 1.0);
            virtual ~TwoLevelRiskyTreasureEnv() = default;

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
