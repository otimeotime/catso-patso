#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>

namespace mcts::exp {
    class TwoPathSspDeceptiveEnv : public mcts::MctsEnv {
        public:
            static constexpr int choose_safe_action = 0;
            static constexpr int choose_risky_action = 1;
            static constexpr int continue_action = 0;

            static constexpr int root_mode = 0;
            static constexpr int safe_mode = 1;
            static constexpr int risky_mode = 2;

            static constexpr int active_status = 0;
            static constexpr int goal_status = 1;
            static constexpr int hazard_status = 2;
            static constexpr int truncated_status = 3;

        private:
            int safe_length;
            int risky_length;
            int hazard_index;
            double hazard_prob;
            double step_cost;
            double goal_reward;
            double hazard_penalty;
            int max_steps;

            void decode_state(
                std::shared_ptr<const mcts::Int4TupleState> state,
                int& mode,
                int& index,
                int& steps,
                int& status) const;
            std::shared_ptr<const mcts::Int4TupleState> make_state(int mode, int index, int steps, int status) const;

        public:
            TwoPathSspDeceptiveEnv(
                int safe_length = 6,
                int risky_length = 2,
                int hazard_index = 0,
                double hazard_prob = 0.08,
                double step_cost = -1.0,
                double goal_reward = 30.0,
                double hazard_penalty = -120.0,
                int max_steps = 100);
            virtual ~TwoPathSspDeceptiveEnv() = default;

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
