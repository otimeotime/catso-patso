#include "env/frozen_lake_env.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "helper_templates.h"

using namespace std;

namespace mcts::exp {
    static pair<int, int> find_unique_cell(const vector<string>& grid, char ch) {
        pair<int, int> found{-1, -1};
        for (int y = 0; y < (int)grid.size(); ++y) {
            for (int x = 0; x < (int)grid[y].size(); ++x) {
                if (grid[y][x] == ch) {
                    if (found.first != -1) {
                        throw runtime_error(string("FrozenLakeEnv: expected exactly one '") + ch + "' cell");
                    }
                    found = {x, y};
                }
            }
        }
        if (found.first == -1) {
            throw runtime_error(string("FrozenLakeEnv: missing required '") + ch + "' cell");
        }
        return found;
    }

        FrozenLakeEnv::FrozenLakeEnv(vector<string> grid, int horizon, double gamma, double intended_move_prob)
                : mcts::MctsEnv(true),
                    height((int)grid.size()),
                    width(grid.empty() ? 0 : (int)grid[0].size()),
                    horizon(horizon),
                    gamma(gamma),
                    intended_move_prob(intended_move_prob),
                    grid(std::move(grid))
    {
        if (height <= 0 || width <= 0) throw runtime_error("FrozenLakeEnv: empty grid");
        for (const auto& row : this->grid) {
            if ((int)row.size() != width) throw runtime_error("FrozenLakeEnv: non-rectangular grid");
        }
        if (horizon <= 0) throw runtime_error("FrozenLakeEnv: horizon must be > 0");
        if (gamma <= 0.0 || gamma > 1.0) throw runtime_error("FrozenLakeEnv: gamma must be in (0,1]");
        if (intended_move_prob < 0.0 || intended_move_prob > 1.0) {
            throw runtime_error("FrozenLakeEnv: intended_move_prob must be in [0,1]");
        }

        start_xy = find_unique_cell(this->grid, 'S');
        goal_xy = find_unique_cell(this->grid, 'G');
    }

    bool FrozenLakeEnv::in_bounds(int x, int y) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    char FrozenLakeEnv::cell_at(int x, int y) const {
        if (!in_bounds(x, y)) return 'W';
        return grid[y][x];
    }

    bool FrozenLakeEnv::is_wall_cell(int x, int y) const {
        return cell_at(x, y) == 'W';
    }

    bool FrozenLakeEnv::is_hole_cell(int x, int y) const {
        return cell_at(x, y) == 'H';
    }

    bool FrozenLakeEnv::is_goal_cell(int x, int y) const {
        return make_pair(x, y) == goal_xy;
    }

    shared_ptr<const mcts::Int3TupleState> FrozenLakeEnv::get_initial_state() const {
        return make_shared<mcts::Int3TupleState>(start_xy.first, start_xy.second, 0);
    }

    bool FrozenLakeEnv::is_sink_state(shared_ptr<const mcts::Int3TupleState> state) const {
        int x = get<0>(state->state);
        int y = get<1>(state->state);
        int t = get<2>(state->state);
        if (t >= horizon) return true;
        if (is_goal_cell(x, y)) return true;
        if (is_hole_cell(x, y)) return true;
        return false;
    }

    shared_ptr<mcts::StringActionVector> FrozenLakeEnv::get_valid_actions(shared_ptr<const mcts::Int3TupleState> state) const {
        auto valid_actions = make_shared<mcts::StringActionVector>();
        if (is_sink_state(state)) return valid_actions;
        valid_actions->push_back(make_shared<const mcts::StringAction>("left"));
        valid_actions->push_back(make_shared<const mcts::StringAction>("right"));
        valid_actions->push_back(make_shared<const mcts::StringAction>("up"));
        valid_actions->push_back(make_shared<const mcts::StringAction>("down"));
        return valid_actions;
    }

    shared_ptr<const mcts::Int3TupleState> FrozenLakeEnv::make_next_state_deterministic(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::StringAction> action) const
    {
        int x = get<0>(state->state);
        int y = get<1>(state->state);
        int t = get<2>(state->state);

        if (is_sink_state(state)) {
            return make_shared<mcts::Int3TupleState>(x, y, t);
        }

        int nx = x;
        int ny = y;
        if (action->action == "left") nx -= 1;
        else if (action->action == "right") nx += 1;
        else if (action->action == "up") ny -= 1;
        else if (action->action == "down") ny += 1;

        if (!in_bounds(nx, ny) || is_wall_cell(nx, ny)) {
            nx = x;
            ny = y;
        }

        int nt = t + 1;
        return make_shared<mcts::Int3TupleState>(nx, ny, nt);
    }

    shared_ptr<mcts::Int3TupleStateDistr> FrozenLakeEnv::get_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::StringAction> action) const
    {
        auto distr = make_shared<mcts::Int3TupleStateDistr>();

        int x = get<0>(state->state);
        int y = get<1>(state->state);
        int t = get<2>(state->state);

        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }

        const int nt = t + 1;

        auto apply_move = [&](int dx, int dy) {
            int nx = x + dx;
            int ny = y + dy;
            if (!in_bounds(nx, ny) || is_wall_cell(nx, ny)) {
                nx = x;
                ny = y;
            }
            return make_shared<mcts::Int3TupleState>(nx, ny, nt);
        };

        // Order: left, right, up, down to match actions above
        auto left_state = apply_move(-1, 0);
        auto right_state = apply_move(1, 0);
        auto up_state = apply_move(0, -1);
        auto down_state = apply_move(0, 1);

        // Build list of slip directions (the 3 directions other than intended)
        vector<shared_ptr<const mcts::Int3TupleState>> slip_states;
        shared_ptr<const mcts::Int3TupleState> intended_state;
        
        if (action->action == "left") {
            intended_state = left_state;
            slip_states = {right_state, up_state, down_state};
        } else if (action->action == "right") {
            intended_state = right_state;
            slip_states = {left_state, up_state, down_state};
        } else if (action->action == "up") {
            intended_state = up_state;
            slip_states = {left_state, right_state, down_state};
        } else if (action->action == "down") {
            intended_state = down_state;
            slip_states = {left_state, right_state, up_state};
        } else {
            throw runtime_error("FrozenLakeEnv: unknown action '" + action->action + "'");
        }

        const double slip_prob = 1.0 - intended_move_prob;
        const double slip_prob_each = slip_prob / 3.0;
        const double epsilon = 1e-15;  // Skip probabilities below this threshold

        auto accumulate = [&](shared_ptr<const mcts::Int3TupleState> s, double p) {
            // Skip negligible probabilities to avoid floating-point accumulation errors
            if (p < epsilon) return;
            
            // Search for existing state by VALUE, not pointer
            for (auto& kv : *distr) {
                if (kv.first->state == s->state) {
                    kv.second += p;
                    return;
                }
            }
            // Not found, insert new entry
            distr->emplace(s, p);
        };

        accumulate(intended_state, intended_move_prob);
        for (const auto& slip_state : slip_states) {
            accumulate(slip_state, slip_prob_each);
        }

        // Normalize probabilities to handle floating-point errors
        // (especially important when p=1.0 or p≈0.0)
        double total_prob = 0.0;
        for (const auto& kv : *distr) {
            total_prob += kv.second;
        }
        if (total_prob > 0.0) {
            for (auto& kv : *distr) {
                kv.second /= total_prob;
            }
        }

        return distr;
    }

    shared_ptr<const mcts::Int3TupleState> FrozenLakeEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::StringAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double FrozenLakeEnv::get_reward(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::StringAction> action,
        shared_ptr<const mcts::Int3TupleState> observation) const
    {
        if (is_sink_state(state)) return 0.0;

        auto goal_reward = [&](shared_ptr<const mcts::Int3TupleState> s) {
            int nx = get<0>(s->state);
            int ny = get<1>(s->state);
            int nt = get<2>(s->state);
            return is_goal_cell(nx, ny) ? pow(gamma, (double)nt) : 0.0;
        };

        if (observation != nullptr) {
            return goal_reward(observation);
        }

        double expected_r = 0.0;
        auto distr = get_transition_distribution(state, action);
        for (const auto& kv : *distr) {
            expected_r += kv.second * goal_reward(kv.first);
        }
        return expected_r;
    }

    // Interface
    shared_ptr<const mcts::State> FrozenLakeEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool FrozenLakeEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const mcts::Int3TupleState>(state);
        return is_sink_state(s);
    }

    shared_ptr<mcts::ActionVector> FrozenLakeEnv::get_valid_actions_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto acts = get_valid_actions(s);
        auto out = make_shared<mcts::ActionVector>();
        for (const auto& a : *acts) out->push_back(static_pointer_cast<const mcts::Action>(a));
        return out;
    }

    shared_ptr<mcts::StateDistr> FrozenLakeEnv::get_transition_distribution_itfc(
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

    shared_ptr<const mcts::State> FrozenLakeEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        auto s = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto a = static_pointer_cast<const mcts::StringAction>(action);
        auto ns = sample_transition_distribution(s, a, rand_manager);
        return static_pointer_cast<const mcts::State>(ns);
    }

    double FrozenLakeEnv::get_reward_itfc(
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