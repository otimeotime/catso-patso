#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcts::exp {
    /**
     * Taxi environment (Taxi-v3 style).
     *
     * State encoding matches Gymnasium Taxi-v3:
     *   ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination
     * with passenger_loc in {0:R, 1:G, 2:Y, 3:B, 4:in-taxi} and destination in {0:R,1:G,2:Y,3:B}.
     *
     * Actions:
     *   0: south, 1: north, 2: east, 3: west, 4: pickup, 5: dropoff.
     */
    class TaxiEnv : public mcts::MctsEnv {
        private:
            static constexpr int kGridSize = 5;
            static constexpr int kNumLocations = 4;
            static constexpr int kPassengerInTaxi = 4;

            static const std::array<std::pair<int, int>, kNumLocations> kLocations;
            static const std::vector<std::string> kMap;

            bool is_raining;
            int initial_state;

            bool has_wall_east(int row, int col) const;
            bool has_wall_west(int row, int col) const;
            std::pair<int, int> move_deterministic(int row, int col, int action) const;

        public:
            TaxiEnv(int initial_state = -1, bool is_raining = false);
            virtual ~TaxiEnv() = default;

            // Helpers
            static int encode(int taxi_row, int taxi_col, int passenger_loc, int destination);
            static void decode(int state, int& taxi_row, int& taxi_col, int& passenger_loc, int& destination);
            static bool is_valid_state(int state);

            void set_initial_state(int state);

            // Typed API
            std::shared_ptr<const mcts::IntState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::IntState> state) const;
            std::shared_ptr<mcts::IntActionVector> get_valid_actions(std::shared_ptr<const mcts::IntState> state) const;
            std::shared_ptr<std::unordered_map<std::shared_ptr<const mcts::IntState>, double>> get_transition_distribution(
                std::shared_ptr<const mcts::IntState> state,
                std::shared_ptr<const mcts::IntAction> action) const;
            std::shared_ptr<const mcts::IntState> sample_transition_distribution(
                std::shared_ptr<const mcts::IntState> state,
                std::shared_ptr<const mcts::IntAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const mcts::IntState> state,
                std::shared_ptr<const mcts::IntAction> action,
                std::shared_ptr<const mcts::IntState> observation = nullptr) const;

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
                std::shared_ptr<const mcts::Observation> observation = nullptr) const;
    };
}
