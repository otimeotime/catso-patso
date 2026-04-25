#pragma once

#include "mcts_env.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace mcts::exp {
    class LunarLanderState : public mcts::State {
        public:
            int x;
            int y;
            int vx;
            int vy;
            int angle;
            int step;
            int status;
            int last_bonus_outcome;

            LunarLanderState(
                int x,
                int y,
                int vx,
                int vy,
                int angle,
                int step,
                int status,
                int last_bonus_outcome)
                : x(x),
                  y(y),
                  vx(vx),
                  vy(vy),
                  angle(angle),
                  step(step),
                  status(status),
                  last_bonus_outcome(last_bonus_outcome) {}
            virtual ~LunarLanderState() = default;

            virtual std::size_t hash() const override;
            bool equals(const LunarLanderState& other) const;
            virtual bool equals_itfc(const mcts::Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    class LunarLanderEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 4;
            static constexpr int no_op_action = 0;
            static constexpr int left_engine_action = 1;
            static constexpr int main_engine_action = 2;
            static constexpr int right_engine_action = 3;

            static constexpr int active_status = 0;
            static constexpr int landed_status = 1;
            static constexpr int crashed_status = 2;
            static constexpr int truncated_status = 3;

            static constexpr int num_bonus_outcomes = 5;
            static constexpr int neutral_bonus_outcome = 2;

        private:
            int max_steps;
            double reward_normalisation;

            using LunarLanderStateDistr = std::unordered_map<std::shared_ptr<const LunarLanderState>, double>;

            std::shared_ptr<const LunarLanderState> make_state(
                int x,
                int y,
                int vx,
                int vy,
                int angle,
                int step,
                int status,
                int last_bonus_outcome) const;
            double landing_bonus_value(int outcome_index) const;
            double landing_bonus_probability(int outcome_index) const;

        public:
            LunarLanderEnv(int max_steps = 200, double reward_normalisation = 256.0);
            virtual ~LunarLanderEnv() = default;

            std::shared_ptr<const LunarLanderState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const LunarLanderState> state) const;
            bool is_catastrophic_state(std::shared_ptr<const LunarLanderState> state) const;
            std::shared_ptr<mcts::IntActionVector> get_valid_actions(
                std::shared_ptr<const LunarLanderState> state) const;
            std::shared_ptr<LunarLanderStateDistr> get_transition_distribution(
                std::shared_ptr<const LunarLanderState> state,
                std::shared_ptr<const mcts::IntAction> action) const;
            std::shared_ptr<const LunarLanderState> sample_transition_distribution(
                std::shared_ptr<const LunarLanderState> state,
                std::shared_ptr<const mcts::IntAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const LunarLanderState> state,
                std::shared_ptr<const mcts::IntAction> action,
                std::shared_ptr<const LunarLanderState> observation = nullptr) const;

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
