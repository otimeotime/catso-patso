#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <array>
#include <memory>
#include <vector>

namespace mcts::exp {
    class AutonomousVehicleEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 4;
            static constexpr int move_positive_y_action = 0;
            static constexpr int move_positive_x_action = 1;
            static constexpr int move_negative_y_action = 2;
            static constexpr int move_negative_x_action = 3;

            static constexpr int num_reward_outcomes = 3;
            static constexpr int initial_outcome_tag = 3;
            static constexpr double default_goal_bonus = 80.0;
            static constexpr double default_reward_normalisation = 256.0;

        private:
            std::vector<std::vector<int>> horizontal_edges;
            std::vector<std::vector<int>> vertical_edges;
            std::vector<std::array<double, num_reward_outcomes>> edge_rewards;
            std::array<double, 2> edge_probs;
            int max_steps;
            double reward_normalisation;
            int max_x;
            int max_y;

            int encode_time_outcome(int time, int outcome) const;
            void decode_state(
                std::shared_ptr<const mcts::Int3TupleState> state,
                int& x,
                int& y,
                int& time,
                int& last_outcome) const;
            std::shared_ptr<const mcts::Int3TupleState> make_state(int x, int y, int time, int outcome) const;
            bool is_action_valid(int x, int y, int action) const;
            std::pair<int, int> next_position(int x, int y, int action) const;
            int get_edge_type(int x, int y, int action) const;
            double reward_for_transition(int edge_type, int outcome, bool reached_goal) const;

        public:
            AutonomousVehicleEnv(
                std::vector<std::vector<int>> horizontal_edges,
                std::vector<std::vector<int>> vertical_edges,
                std::vector<std::array<double, num_reward_outcomes>> edge_rewards,
                std::array<double, 2> edge_probs,
                int max_steps,
                double reward_normalisation = default_reward_normalisation);
            virtual ~AutonomousVehicleEnv() = default;

            static std::vector<std::vector<int>> default_horizontal_edges();
            static std::vector<std::vector<int>> default_vertical_edges();
            static std::vector<std::array<double, num_reward_outcomes>> default_edge_rewards();
            static std::array<double, 2> default_edge_probs();

            // Typed API
            std::shared_ptr<const mcts::Int3TupleState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::Int3TupleState> state) const;
            bool is_goal_state(std::shared_ptr<const mcts::Int3TupleState> state) const;
            bool is_catastrophic_state(std::shared_ptr<const mcts::Int3TupleState> state) const;
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

            // Interface API
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

            int get_width() const { return max_x; }
            int get_height() const { return max_y; }
            int get_max_steps() const { return max_steps; }
            double get_reward_normalisation() const { return reward_normalisation; }
    };
}
