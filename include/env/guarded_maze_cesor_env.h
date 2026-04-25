#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mcts::exp {
    class GuardedMazeCesorEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 4;
            static constexpr int left_action = 0;
            static constexpr int right_action = 1;
            static constexpr int down_action = 2;
            static constexpr int up_action = 3;

            static constexpr int normal_event = 0;
            static constexpr int blocked_event = 1;
            static constexpr int spotted_event = 2;

        private:
            int mode;
            int rows;
            int cols;
            double action_noise;
            int max_steps;
            double guard_prob;
            double guard_cost;
            double force_motion;
            double reward_normalisation;
            int deduction_limit;
            int L;
            std::array<int, 2> start;
            std::array<int, 2> goal;
            std::vector<std::vector<int>> grid;
            std::vector<std::unordered_map<int, double>> movement_cache;

            int encode_tag(bool spotted, int last_event) const;
            void decode_tag(int tag, bool& spotted, int& last_event) const;
            void decode_state(
                std::shared_ptr<const mcts::Int4TupleState> state,
                int& row,
                int& col,
                int& steps,
                bool& spotted,
                int& last_event) const;
            std::shared_ptr<const mcts::Int4TupleState> make_state(
                int row,
                int col,
                int steps,
                bool spotted,
                int last_event) const;
            bool is_wall(int row, int col) const;
            bool is_goal_cell(int row, int col) const;
            int kill_zone_strength(int row, int col) const;
            int encode_movement_outcome(int row, int col, bool blocked, bool kill_zone) const;
            void decode_movement_outcome(int code, int& row, int& col, bool& blocked, bool& kill_zone) const;
            void build_layout();
            void precompute_movement_cache(int seed, int samples_per_pair);

        public:
            GuardedMazeCesorEnv(
                int mode = 1,
                int rows = -1,
                int cols = -1,
                double action_noise = 0.2,
                int max_steps = -1,
                double guard_prob = 0.05,
                double guard_cost = 4.0,
                double force_motion = 0.0,
                double reward_normalisation = 256.0,
                int deduction_limit = 32,
                std::array<int, 2> start = {1, 2},
                std::array<int, 2> goal = {-1, -1},
                int seed = 1,
                int movement_samples = 4000);
            virtual ~GuardedMazeCesorEnv() = default;

            std::shared_ptr<const mcts::Int4TupleState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::Int4TupleState> state) const;
            bool is_catastrophic_state(std::shared_ptr<const mcts::Int4TupleState> state) const;
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
