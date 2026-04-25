#pragma once

#include "mcts_env.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mcts::exp {
    class ServersSimState : public mcts::State {
        public:
            int paid_servers;
            int booting_one;
            int booting_two;
            int queue_units;
            int peak_steps_remaining;
            int step;

            ServersSimState(
                int paid_servers,
                int booting_one,
                int booting_two,
                int queue_units,
                int peak_steps_remaining,
                int step)
                : paid_servers(paid_servers),
                  booting_one(booting_one),
                  booting_two(booting_two),
                  queue_units(queue_units),
                  peak_steps_remaining(peak_steps_remaining),
                  step(step) {}
            virtual ~ServersSimState() = default;

            virtual std::size_t hash() const override;
            bool equals(const ServersSimState& other) const;
            virtual bool equals_itfc(const mcts::Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    class ServersSimEnv : public mcts::MctsEnv {
        public:
            static constexpr int num_actions = 3;
            static constexpr int keep_action = 0;
            static constexpr int add_server_action = 1;
            static constexpr int remove_server_action = 2;

        private:
            int max_servers;
            int min_servers;
            int init_servers;
            int decision_horizon;
            int peak_duration_steps;
            int queue_unit_size;
            int service_units_per_server;
            int max_queue_units;
            int max_arrival_units;
            int catastrophic_queue_units;
            double normal_arrival_units_mean;
            double peak_arrival_units_mean;
            double peak_start_prob;
            double tts_cost;
            double server_cost;
            double reward_normalisation;
            std::vector<std::pair<int, double>> normal_arrival_distribution;
            std::vector<std::pair<int, double>> peak_arrival_distribution;

            using ServersSimStateDistr = std::unordered_map<std::shared_ptr<const ServersSimState>, double>;

            std::shared_ptr<const ServersSimState> make_state(
                int paid_servers,
                int booting_one,
                int booting_two,
                int queue_units,
                int peak_steps_remaining,
                int step) const;
            void validate_state_components(
                int paid_servers,
                int booting_one,
                int booting_two,
                int queue_units,
                int peak_steps_remaining,
                int step) const;
            std::vector<std::pair<int, double>> build_truncated_poisson(double mean_units) const;
            int active_servers(std::shared_ptr<const ServersSimState> state) const;
            double step_cost(std::shared_ptr<const ServersSimState> observation) const;

        public:
            ServersSimEnv(
                int max_servers = 10,
                int min_servers = 3,
                int init_servers = 4,
                int decision_horizon = 60,
                int peak_duration_steps = 5,
                int queue_unit_size = 20,
                int service_units_per_server = 3,
                int max_queue_units = 40,
                int max_arrival_units = 24,
                double normal_arrival_units_mean = 9.0,
                double peak_arrival_units_mean = 18.0,
                double peak_start_prob = 60.0 / (3.0 * 24.0 * 60.0 * 60.0),
                double tts_cost = 1.0,
                double server_cost = 2.0,
                double reward_normalisation = 1.0);
            virtual ~ServersSimEnv() = default;

            std::shared_ptr<const ServersSimState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const ServersSimState> state) const;
            bool is_catastrophic_state(std::shared_ptr<const ServersSimState> state) const;
            std::shared_ptr<mcts::IntActionVector> get_valid_actions(
                std::shared_ptr<const ServersSimState> state) const;
            std::shared_ptr<ServersSimStateDistr> get_transition_distribution(
                std::shared_ptr<const ServersSimState> state,
                std::shared_ptr<const mcts::IntAction> action) const;
            std::shared_ptr<const ServersSimState> sample_transition_distribution(
                std::shared_ptr<const ServersSimState> state,
                std::shared_ptr<const mcts::IntAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const ServersSimState> state,
                std::shared_ptr<const mcts::IntAction> action,
                std::shared_ptr<const ServersSimState> observation = nullptr) const;

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
