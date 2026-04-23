#include "env/sailing_env.h"

#include <cmath>
#include <memory>
#include <stdexcept>

using namespace std;

namespace mcts::exp {
    SailingEnv::SailingEnv(
        int grid_size,
        int init_wind_dir,
        double wind_stay_prob,
        double wind_turn_prob,
        double base_cost,
        double angle_cost_scale,
        double goal_reward)
        : mcts::MctsEnv(true),
          grid_size(grid_size),
          init_wind_dir(init_wind_dir),
          wind_stay_prob(wind_stay_prob),
          wind_turn_prob(wind_turn_prob),
          base_cost(base_cost),
          angle_cost_scale(angle_cost_scale),
          goal_reward(goal_reward)
    {
        if (grid_size <= 1) throw runtime_error("SailingEnv: grid_size must be > 1");
        if (init_wind_dir < 0 || init_wind_dir > 7) throw runtime_error("SailingEnv: init_wind_dir must be in [0,7]");
        if (wind_stay_prob < 0.0 || wind_turn_prob < 0.0) throw runtime_error("SailingEnv: wind probabilities must be non-negative");
        if (abs((wind_stay_prob + 2.0 * wind_turn_prob) - 1.0) > 1e-9) {
            throw runtime_error("SailingEnv: expected wind_stay_prob + 2*wind_turn_prob == 1");
        }
        if (base_cost <= 0.0) throw runtime_error("SailingEnv: base_cost must be > 0");
        if (angle_cost_scale < 0.0) throw runtime_error("SailingEnv: angle_cost_scale must be >= 0");
    }

    bool SailingEnv::in_bounds(int x, int y) const {
        return x >= 0 && x < grid_size && y >= 0 && y < grid_size;
    }

    int SailingEnv::action_dir_index(const string& action) const {
        // 0:E, 1:NE, 2:N, 3:NW, 4:W, 5:SW, 6:S, 7:SE
        if (action == "E") return 0;
        if (action == "NE") return 1;
        if (action == "N") return 2;
        if (action == "NW") return 3;
        if (action == "W") return 4;
        if (action == "SW") return 5;
        if (action == "S") return 6;
        if (action == "SE") return 7;
        throw runtime_error("SailingEnv: unknown action: " + action);
    }

    string SailingEnv::dir_index_to_action(int dir) const {
        dir = (dir % 8 + 8) % 8;
        switch (dir) {
            case 0: return "E";
            case 1: return "NE";
            case 2: return "N";
            case 3: return "NW";
            case 4: return "W";
            case 5: return "SW";
            case 6: return "S";
            case 7: return "SE";
            default: return "E";
        }
    }

    pair<int, int> SailingEnv::move_delta_from_action(const string& action) const {
        int dir = action_dir_index(action);
        switch (dir) {
            case 0: return {+1, 0};
            case 1: return {+1, -1};
            case 2: return {0, -1};
            case 3: return {-1, -1};
            case 4: return {-1, 0};
            case 5: return {-1, +1};
            case 6: return {0, +1};
            case 7: return {+1, +1};
            default: return {0, 0};
        }
    }

    bool SailingEnv::is_into_wind(int action_dir, int wind_dir) const {
        // Interpret wind_dir as the direction the wind is blowing *towards*.
        // Sailing "into the wind" means heading directly opposite to wind_dir.
        int diff = (action_dir - wind_dir + 8) % 8;
        return diff == 4;
    }

    double SailingEnv::compute_cost(int action_dir, int wind_dir) const {
        // Angle between travel direction and wind direction (0..180 degrees).
        int raw = (action_dir - wind_dir + 8) % 8;
        int steps = raw;
        if (steps > 4) steps = 8 - steps;
        double angle = 45.0 * (double)steps;
        // Penalize sailing away from downwind; 0 deg is easiest, 180 deg is invalid (filtered).
        return base_cost + angle_cost_scale * (angle / 90.0);
    }

    shared_ptr<const mcts::Int3TupleState> SailingEnv::get_initial_state() const {
        return make_shared<mcts::Int3TupleState>(0, 0, init_wind_dir);
    }

    bool SailingEnv::is_sink_state(shared_ptr<const mcts::Int3TupleState> state) const {
        int x = get<0>(state->state);
        int y = get<1>(state->state);
        return x == (grid_size - 1) && y == (grid_size - 1);
    }

    shared_ptr<mcts::StringActionVector> SailingEnv::get_valid_actions(shared_ptr<const mcts::Int3TupleState> state) const {
        auto valid_actions = make_shared<mcts::StringActionVector>();
        if (is_sink_state(state)) return valid_actions;

        int x = get<0>(state->state);
        int y = get<1>(state->state);
        int wind_dir = get<2>(state->state);

        for (int dir = 0; dir < 8; ++dir) {
            if (is_into_wind(dir, wind_dir)) continue;
            auto act_str = dir_index_to_action(dir);
            auto dxy = move_delta_from_action(act_str);
            int nx = x + dxy.first;
            int ny = y + dxy.second;
            if (!in_bounds(nx, ny)) continue;
            valid_actions->push_back(make_shared<const mcts::StringAction>(act_str));
        }
        return valid_actions;
    }

    shared_ptr<mcts::Int3TupleStateDistr> SailingEnv::get_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::StringAction> action) const
    {
        auto distr = make_shared<mcts::Int3TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(make_shared<mcts::Int3TupleState>(state->state), 1.0);
            return distr;
        }

        int x = get<0>(state->state);
        int y = get<1>(state->state);
        int wind_dir = get<2>(state->state);

        auto dxy = move_delta_from_action(action->action);
        int nx = x + dxy.first;
        int ny = y + dxy.second;
        if (!in_bounds(nx, ny)) {
            nx = x;
            ny = y;
        }

        int left = (wind_dir + 1) % 8;
        int right = (wind_dir + 7) % 8;
        int stay = wind_dir;

        distr->insert_or_assign(make_shared<mcts::Int3TupleState>(nx, ny, stay), wind_stay_prob);
        if (wind_turn_prob > 0.0) {
            distr->insert_or_assign(make_shared<mcts::Int3TupleState>(nx, ny, left), wind_turn_prob);
            distr->insert_or_assign(make_shared<mcts::Int3TupleState>(nx, ny, right), wind_turn_prob);
        }

        return distr;
    }

    shared_ptr<const mcts::Int3TupleState> SailingEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::StringAction> action,
        mcts::RandManager& rand_manager) const
    {
        if (is_sink_state(state)) return make_shared<mcts::Int3TupleState>(state->state);

        int x = get<0>(state->state);
        int y = get<1>(state->state);
        int wind_dir = get<2>(state->state);

        auto dxy = move_delta_from_action(action->action);
        int nx = x + dxy.first;
        int ny = y + dxy.second;
        if (!in_bounds(nx, ny)) {
            nx = x;
            ny = y;
        }

        double u = rand_manager.get_rand_uniform();
        int new_wind = wind_dir;
        if (u < wind_turn_prob) {
            new_wind = (wind_dir + 1) % 8;
        } else if (u < 2.0 * wind_turn_prob) {
            new_wind = (wind_dir + 7) % 8;
        }

        return make_shared<mcts::Int3TupleState>(nx, ny, new_wind);
    }

    double SailingEnv::get_reward(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::StringAction> action,
        shared_ptr<const mcts::Int3TupleState> /*observation*/) const
    {
        if (is_sink_state(state)) return goal_reward;
        int wind_dir = get<2>(state->state);
        int action_dir = action_dir_index(action->action);
        return -compute_cost(action_dir, wind_dir);
    }

    // Interface
    shared_ptr<const mcts::State> SailingEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool SailingEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const mcts::Int3TupleState>(state);
        return is_sink_state(s);
    }

    shared_ptr<mcts::ActionVector> SailingEnv::get_valid_actions_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto acts = get_valid_actions(s);
        auto out = make_shared<mcts::ActionVector>();
        for (const auto& a : *acts) out->push_back(static_pointer_cast<const mcts::Action>(a));
        return out;
    }

    shared_ptr<mcts::StateDistr> SailingEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto s = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto a = static_pointer_cast<const mcts::StringAction>(action);
        auto distr_typed = get_transition_distribution(s, a);
        auto out = make_shared<mcts::StateDistr>();
        for (auto& kv : *distr_typed) {
            out->insert_or_assign(static_pointer_cast<const mcts::State>(kv.first), kv.second);
        }
        return out;
    }

    shared_ptr<const mcts::State> SailingEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        auto s = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto a = static_pointer_cast<const mcts::StringAction>(action);
        auto ns = sample_transition_distribution(s, a, rand_manager);
        return static_pointer_cast<const mcts::State>(ns);
    }

    double SailingEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        auto s = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto a = static_pointer_cast<const mcts::StringAction>(action);
        auto o = static_pointer_cast<const mcts::Int3TupleState>(observation);
        return get_reward(s, a, o);
    }
}