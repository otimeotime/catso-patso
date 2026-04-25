#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace mcts::exp {
    class BettingGameState : public mcts::State {
        public:
            double bankroll;
            int time;

            BettingGameState(double bankroll, int time) : bankroll(bankroll), time(time) {}
            virtual ~BettingGameState() = default;

            virtual std::size_t hash() const override;
            bool equals(const BettingGameState& other) const;
            virtual bool equals_itfc(const mcts::Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    class BettingGameEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 9;
            static constexpr double default_max_state_value = 256.0;
            static constexpr double default_initial_state = 16.0;
            static constexpr double default_reward_normalisation = 256.0;

        private:
            double win_prob;
            int max_sequence_length;
            double max_state_value;
            double initial_state;
            double reward_normalisation;

            std::shared_ptr<const BettingGameState> make_state(double bankroll, int time) const;
            double compute_bet(
                std::shared_ptr<const BettingGameState> state,
                std::shared_ptr<const mcts::IntAction> action) const;
            bool is_absorbing_bankroll(double bankroll) const;

        public:
            BettingGameEnv(
                double win_prob,
                int max_sequence_length,
                double max_state_value = default_max_state_value,
                double initial_state = default_initial_state,
                double reward_normalisation = default_reward_normalisation);
            virtual ~BettingGameEnv() = default;

            using BettingGameStateDistr = std::unordered_map<std::shared_ptr<const BettingGameState>, double>;

            // Typed API
            std::shared_ptr<const BettingGameState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const BettingGameState> state) const;
            bool is_catastrophic_state(std::shared_ptr<const BettingGameState> state) const;
            std::shared_ptr<mcts::IntActionVector> get_valid_actions(
                std::shared_ptr<const BettingGameState> state) const;
            std::shared_ptr<BettingGameStateDistr> get_transition_distribution(
                std::shared_ptr<const BettingGameState> state,
                std::shared_ptr<const mcts::IntAction> action) const;
            std::shared_ptr<const BettingGameState> sample_transition_distribution(
                std::shared_ptr<const BettingGameState> state,
                std::shared_ptr<const mcts::IntAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const BettingGameState> state,
                std::shared_ptr<const mcts::IntAction> action,
                std::shared_ptr<const BettingGameState> observation = nullptr) const;

            // Interface API
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

            double get_win_prob() const { return win_prob; }
            int get_max_sequence_length() const { return max_sequence_length; }
            double get_max_state_value() const { return max_state_value; }
            double get_reward_normalisation() const { return reward_normalisation; }
    };
}
