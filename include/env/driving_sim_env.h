#pragma once

#include "mcts_env.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

namespace mcts::exp {
    class DrivingSimState : public mcts::State {
        public:
            int headway;
            int lane_offset;
            int speed_diff;
            int step;
            int prev_acc_tag;
            int status;

            DrivingSimState(
                int headway,
                int lane_offset,
                int speed_diff,
                int step,
                int prev_acc_tag,
                int status)
                : headway(headway),
                  lane_offset(lane_offset),
                  speed_diff(speed_diff),
                  step(step),
                  prev_acc_tag(prev_acc_tag),
                  status(status) {}
            virtual ~DrivingSimState() = default;

            virtual std::size_t hash() const override;
            bool equals(const DrivingSimState& other) const;
            virtual bool equals_itfc(const mcts::Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    class DrivingSimEnv : public mcts::MctsEnv {
        public:
            static constexpr int active_status = 0;
            static constexpr int collision_status = 1;
            static constexpr int escaped_status = 2;
            static constexpr int truncated_status = 3;

        private:
            bool nine_actions;
            int max_steps;
            int max_headway;
            int max_lane_offset;
            int max_speed_diff;
            double p_oil;
            double reward_normalisation;
            std::array<double, 5> leader_probs;
            std::array<std::pair<int, int>, 9> agent_actions_map;
            int num_actions;

            using DrivingSimStateDistr = std::unordered_map<std::shared_ptr<const DrivingSimState>, double>;

            std::shared_ptr<const DrivingSimState> make_state(
                int headway,
                int lane_offset,
                int speed_diff,
                int step,
                int prev_acc_tag,
                int status) const;
            int clamp_headway(int value) const;
            int clamp_lane(int value) const;
            int clamp_speed_diff(int value) const;
            int encode_acc_tag(int acceleration_delta) const;

        public:
            DrivingSimEnv(
                bool nine_actions = false,
                int max_steps = 60,
                int max_headway = 20,
                int max_lane_offset = 4,
                int max_speed_diff = 6,
                double p_oil = 0.0,
                std::array<double, 5> leader_probs = {0.3, 0.3, 0.3, 0.0, 0.1},
                double reward_normalisation = 256.0);
            virtual ~DrivingSimEnv() = default;

            std::shared_ptr<const DrivingSimState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const DrivingSimState> state) const;
            bool is_catastrophic_state(std::shared_ptr<const DrivingSimState> state) const;
            std::shared_ptr<mcts::IntActionVector> get_valid_actions(
                std::shared_ptr<const DrivingSimState> state) const;
            std::shared_ptr<DrivingSimStateDistr> get_transition_distribution(
                std::shared_ptr<const DrivingSimState> state,
                std::shared_ptr<const mcts::IntAction> action) const;
            std::shared_ptr<const DrivingSimState> sample_transition_distribution(
                std::shared_ptr<const DrivingSimState> state,
                std::shared_ptr<const mcts::IntAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const DrivingSimState> state,
                std::shared_ptr<const mcts::IntAction> action,
                std::shared_ptr<const DrivingSimState> observation = nullptr) const;

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
