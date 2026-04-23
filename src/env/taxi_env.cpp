#include "env/taxi_env.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "helper_templates.h"

using namespace std;

namespace mcts::exp {
    const array<pair<int, int>, TaxiEnv::kNumLocations> TaxiEnv::kLocations = {
        pair<int, int>{0, 0},  // R
        pair<int, int>{0, 4},  // G
        pair<int, int>{4, 0},  // Y
        pair<int, int>{4, 3}   // B
    };

    const vector<string> TaxiEnv::kMap = {
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |"
    };

    TaxiEnv::TaxiEnv(int initial_state, bool is_raining)
        : mcts::MctsEnv(true),
          is_raining(is_raining),
          initial_state(-1)
    {
        if (initial_state != -1) {
            set_initial_state(initial_state);
        }
    }

    int TaxiEnv::encode(int taxi_row, int taxi_col, int passenger_loc, int destination) {
        return ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination;
    }

    void TaxiEnv::decode(int state, int& taxi_row, int& taxi_col, int& passenger_loc, int& destination) {
        if (state < 0) throw runtime_error("TaxiEnv: invalid state (negative)");
        destination = state % 4;
        state /= 4;
        passenger_loc = state % 5;
        state /= 5;
        taxi_col = state % 5;
        state /= 5;
        taxi_row = state;
    }

    bool TaxiEnv::is_valid_state(int state) {
        if (state < 0 || state >= 500) return false;
        int taxi_row = 0;
        int taxi_col = 0;
        int passenger_loc = 0;
        int destination = 0;
        decode(state, taxi_row, taxi_col, passenger_loc, destination);
        if (taxi_row < 0 || taxi_row >= kGridSize) return false;
        if (taxi_col < 0 || taxi_col >= kGridSize) return false;
        if (passenger_loc < 0 || passenger_loc > kPassengerInTaxi) return false;
        if (destination < 0 || destination >= kNumLocations) return false;
        return true;
    }

    void TaxiEnv::set_initial_state(int state) {
        if (!is_valid_state(state)) {
            throw runtime_error("TaxiEnv: initial_state must be in [0,499]");
        }
        initial_state = state;
    }

    bool TaxiEnv::has_wall_east(int row, int col) const {
        if (row < 0 || row >= kGridSize || col < 0 || col >= kGridSize) return true;
        return kMap[row][2 * col + 2] == '|';
    }

    bool TaxiEnv::has_wall_west(int row, int col) const {
        if (row < 0 || row >= kGridSize || col < 0 || col >= kGridSize) return true;
        return kMap[row][2 * col] == '|';
    }

    pair<int, int> TaxiEnv::move_deterministic(int row, int col, int action) const {
        int nr = row;
        int nc = col;

        if (action == 0) {  // south
            if (row < kGridSize - 1) nr = row + 1;
        } else if (action == 1) {  // north
            if (row > 0) nr = row - 1;
        } else if (action == 2) {  // east
            if (!has_wall_east(row, col) && col < kGridSize - 1) nc = col + 1;
        } else if (action == 3) {  // west
            if (!has_wall_west(row, col) && col > 0) nc = col - 1;
        }

        return {nr, nc};
    }

    shared_ptr<const mcts::IntState> TaxiEnv::get_initial_state() const {
        if (initial_state >= 0) {
            return make_shared<mcts::IntState>(initial_state);
        }

        int state = encode(0, 0, 0, 1);
        return make_shared<mcts::IntState>(state);
    }

    bool TaxiEnv::is_sink_state(shared_ptr<const mcts::IntState> state) const {
        int taxi_row = 0;
        int taxi_col = 0;
        int passenger_loc = 0;
        int destination = 0;
        decode(state->state, taxi_row, taxi_col, passenger_loc, destination);
        if (passenger_loc == kPassengerInTaxi) return false;
        return passenger_loc == destination;
    }

    shared_ptr<mcts::IntActionVector> TaxiEnv::get_valid_actions(shared_ptr<const mcts::IntState> state) const {
        auto valid_actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) return valid_actions;
        for (int a = 0; a < 6; ++a) {
            valid_actions->push_back(make_shared<const mcts::IntAction>(a));
        }
        return valid_actions;
    }

    shared_ptr<unordered_map<shared_ptr<const mcts::IntState>, double>> TaxiEnv::get_transition_distribution(
        shared_ptr<const mcts::IntState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<unordered_map<shared_ptr<const mcts::IntState>, double>>();

        int taxi_row = 0;
        int taxi_col = 0;
        int passenger_loc = 0;
        int destination = 0;
        decode(state->state, taxi_row, taxi_col, passenger_loc, destination);

        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }

        if (action->action <= 3) {
            vector<pair<int, int>> candidates;
            candidates.reserve(3);
            candidates.push_back(move_deterministic(taxi_row, taxi_col, action->action));

            if (is_raining) {
                int left_action = (action->action == 0) ? 2 : (action->action == 1) ? 3 : (action->action == 2) ? 1 : 0;
                int right_action = (action->action == 0) ? 3 : (action->action == 1) ? 2 : (action->action == 2) ? 0 : 1;
                candidates.push_back(move_deterministic(taxi_row, taxi_col, left_action));
                candidates.push_back(move_deterministic(taxi_row, taxi_col, right_action));
            }

            vector<double> probs;
            if (is_raining) {
                probs = {0.9, 0.05, 0.05};
            } else {
                probs = {1.0};
            }

            for (size_t i = 0; i < candidates.size(); ++i) {
                int nr = candidates[i].first;
                int nc = candidates[i].second;
                int next_state = encode(nr, nc, passenger_loc, destination);
                auto ns = make_shared<mcts::IntState>(next_state);

                bool merged = false;
                for (auto& kv : *distr) {
                    if (kv.first->state == ns->state) {
                        kv.second += probs[i];
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    distr->emplace(ns, probs[i]);
                }
            }
            return distr;
        }

        int next_passenger_loc = passenger_loc;
        int next_taxi_row = taxi_row;
        int next_taxi_col = taxi_col;

        if (action->action == 4) {  // pickup
            if (passenger_loc != kPassengerInTaxi) {
                if (make_pair(taxi_row, taxi_col) == kLocations[passenger_loc]) {
                    next_passenger_loc = kPassengerInTaxi;
                }
            }
        } else if (action->action == 5) {  // dropoff
            if (passenger_loc == kPassengerInTaxi) {
                if (make_pair(taxi_row, taxi_col) == kLocations[destination]) {
                    next_passenger_loc = destination;
                }
            }
        }

        int next_state = encode(next_taxi_row, next_taxi_col, next_passenger_loc, destination);
        auto ns = make_shared<mcts::IntState>(next_state);
        distr->insert_or_assign(ns, 1.0);
        return distr;
    }

    shared_ptr<const mcts::IntState> TaxiEnv::sample_transition_distribution(
        shared_ptr<const mcts::IntState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double TaxiEnv::get_reward(
        shared_ptr<const mcts::IntState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const mcts::IntState> observation) const
    {
        if (is_sink_state(state)) return 0.0;

        int taxi_row = 0;
        int taxi_col = 0;
        int passenger_loc = 0;
        int destination = 0;
        decode(state->state, taxi_row, taxi_col, passenger_loc, destination);

        if (action->action == 4) {  // pickup
            if (passenger_loc != kPassengerInTaxi &&
                make_pair(taxi_row, taxi_col) == kLocations[passenger_loc]) {
                return -1.0;
            }
            return -10.0;
        }

        if (action->action == 5) {  // dropoff
            if (passenger_loc == kPassengerInTaxi &&
                make_pair(taxi_row, taxi_col) == kLocations[destination]) {
                return 20.0;
            }
            return -10.0;
        }

        return -1.0;
    }

    shared_ptr<const mcts::State> TaxiEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool TaxiEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const mcts::IntState>(state);
        return is_sink_state(s);
    }

    shared_ptr<mcts::ActionVector> TaxiEnv::get_valid_actions_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const mcts::IntState>(state);
        auto acts = get_valid_actions(s);
        auto out = make_shared<mcts::ActionVector>();
        for (const auto& a : *acts) out->push_back(static_pointer_cast<const mcts::Action>(a));
        return out;
    }

    shared_ptr<mcts::StateDistr> TaxiEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto s = static_pointer_cast<const mcts::IntState>(state);
        auto a = static_pointer_cast<const mcts::IntAction>(action);
        auto distr_typed = get_transition_distribution(s, a);
        auto out = make_shared<mcts::StateDistr>();
        for (auto& kv : *distr_typed) {
            out->insert_or_assign(static_pointer_cast<const mcts::State>(kv.first), kv.second);
        }
        return out;
    }

    shared_ptr<const mcts::State> TaxiEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        auto s = static_pointer_cast<const mcts::IntState>(state);
        auto a = static_pointer_cast<const mcts::IntAction>(action);
        auto ns = sample_transition_distribution(s, a, rand_manager);
        return static_pointer_cast<const mcts::State>(ns);
    }

    double TaxiEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        auto s = static_pointer_cast<const mcts::IntState>(state);
        auto a = static_pointer_cast<const mcts::IntAction>(action);
        auto o = static_pointer_cast<const mcts::IntState>(observation);
        return get_reward(s, a, o);
    }
}
