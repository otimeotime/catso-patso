#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>
#include <string>
#include <utility>

namespace mcts::exp {
    class GridEnv : public mcts::MctsEnv {
        int grid_size;
        double stay_prob;

        std::shared_ptr<const mcts::IntPairState> make_candidate_next_state(
            std::shared_ptr<const mcts::IntPairState> state,
            std::shared_ptr<const mcts::StringAction> action) const;

        public:
            GridEnv(int grid_size, double stay_prob=0.0) : mcts::MctsEnv(true), grid_size(grid_size), stay_prob(stay_prob) {}
            virtual ~GridEnv() = default;

            // Typed API
            std::shared_ptr<const mcts::IntPairState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::IntPairState> state) const;
            std::shared_ptr<mcts::StringActionVector> get_valid_actions(std::shared_ptr<const mcts::IntPairState> state) const;
            std::shared_ptr<mcts::IntPairStateDistr> get_transition_distribution(
                std::shared_ptr<const mcts::IntPairState> state,
                std::shared_ptr<const mcts::StringAction> action) const;
            std::shared_ptr<const mcts::IntPairState> sample_transition_distribution(
                std::shared_ptr<const mcts::IntPairState> state,
                std::shared_ptr<const mcts::StringAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const mcts::IntPairState> state,
                std::shared_ptr<const mcts::StringAction> action,
                std::shared_ptr<const mcts::IntPairState> observation=nullptr) const;

            // Interface API
            virtual std::shared_ptr<const mcts::State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const mcts::State> state) const;
            virtual std::shared_ptr<mcts::ActionVector> get_valid_actions_itfc(std::shared_ptr<const mcts::State> state) const;
            virtual std::shared_ptr<mcts::StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action) const;
            virtual std::shared_ptr<const mcts::State> sample_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                mcts::RandManager& rand_manager) const;
            virtual double get_reward_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                std::shared_ptr<const mcts::Observation> observation=nullptr) const;
    };
}