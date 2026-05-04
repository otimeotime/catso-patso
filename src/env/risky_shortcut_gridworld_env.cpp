#include "env/risky_shortcut_gridworld_env.h"

#include "helper_templates.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>

using namespace std;

namespace mcts::exp {
    RiskyShortcutGridworldEnv::RiskyShortcutGridworldEnv(
        int grid_size,
        double slip_prob,
        double wind_prob,
        vector<int> windy_cols,
        double step_cost,
        double goal_reward,
        double cliff_penalty,
        int max_steps)
        : mcts::MctsEnv(true),
          grid_size(grid_size),
          max_steps(max_steps),
          slip_prob(slip_prob),
          wind_prob(wind_prob),
          step_cost(step_cost),
          goal_reward(goal_reward),
          cliff_penalty(cliff_penalty),
          windy_cols()
    {
        if (grid_size < 4) {
            throw runtime_error("RiskyShortcutGridworldEnv: grid_size must be >= 4");
        }
        if (slip_prob < 0.0 || slip_prob > 1.0) {
            throw runtime_error("RiskyShortcutGridworldEnv: slip_prob must be in [0,1]");
        }
        if (wind_prob < 0.0 || wind_prob > 1.0) {
            throw runtime_error("RiskyShortcutGridworldEnv: wind_prob must be in [0,1]");
        }
        if (this->max_steps < 0) {
            this->max_steps = 10 * grid_size * grid_size;
        }
        if (this->max_steps <= 0) {
            throw runtime_error("RiskyShortcutGridworldEnv: max_steps must be > 0");
        }

        if (windy_cols.empty()) {
            for (int col = 1; col < grid_size - 1; ++col) {
                this->windy_cols.insert(col);
            }
        }
        else {
            for (int col : windy_cols) {
                if (col < 0 || col >= grid_size) {
                    throw runtime_error("RiskyShortcutGridworldEnv: windy column out of bounds");
                }
                this->windy_cols.insert(col);
            }
        }
    }

    shared_ptr<const mcts::Int3TupleState> RiskyShortcutGridworldEnv::make_state(int row, int col, int time) const {
        return make_shared<mcts::Int3TupleState>(row, col, time);
    }

    void RiskyShortcutGridworldEnv::decode_state(
        shared_ptr<const mcts::Int3TupleState> state,
        int& row,
        int& col,
        int& time) const
    {
        row = get<0>(state->state);
        col = get<1>(state->state);
        time = get<2>(state->state);
    }

    bool RiskyShortcutGridworldEnv::is_action_valid(int row, int col, int action) const {
        if (action == up_action) {
            return row > 0;
        }
        if (action == right_action) {
            return col < grid_size - 1;
        }
        if (action == down_action) {
            return row < grid_size - 1;
        }
        if (action == left_action) {
            return col > 0;
        }
        return false;
    }

    pair<int, int> RiskyShortcutGridworldEnv::clip(int row, int col) const {
        return {
            max(0, min(grid_size - 1, row)),
            max(0, min(grid_size - 1, col))
        };
    }

    pair<int, int> RiskyShortcutGridworldEnv::apply_action_deterministic(int row, int col, int action) const {
        int next_row = row;
        int next_col = col;
        if (action == up_action) {
            next_row -= 1;
        }
        else if (action == right_action) {
            next_col += 1;
        }
        else if (action == down_action) {
            next_row += 1;
        }
        else if (action == left_action) {
            next_col -= 1;
        }
        else {
            throw runtime_error("RiskyShortcutGridworldEnv: invalid action");
        }
        return clip(next_row, next_col);
    }

    bool RiskyShortcutGridworldEnv::is_windy_col(int col) const {
        return windy_cols.find(col) != windy_cols.end();
    }

    shared_ptr<const mcts::Int3TupleState> RiskyShortcutGridworldEnv::get_initial_state() const {
        return make_state(grid_size - 1, 0, 0);
    }

    bool RiskyShortcutGridworldEnv::is_goal_cell(int row, int col) const {
        return row == grid_size - 1 && col == grid_size - 1;
    }

    bool RiskyShortcutGridworldEnv::is_cliff_cell(int row, int col) const {
        return row == grid_size - 1 && col > 0 && col < grid_size - 1;
    }

    bool RiskyShortcutGridworldEnv::is_catastrophic_state(shared_ptr<const mcts::Int3TupleState> state) const {
        int row = 0;
        int col = 0;
        int time = 0;
        decode_state(state, row, col, time);
        (void)time;
        return is_cliff_cell(row, col);
    }

    bool RiskyShortcutGridworldEnv::is_sink_state(shared_ptr<const mcts::Int3TupleState> state) const {
        int row = 0;
        int col = 0;
        int time = 0;
        decode_state(state, row, col, time);
        return time >= max_steps || is_goal_cell(row, col) || is_cliff_cell(row, col);
    }

    shared_ptr<mcts::IntActionVector> RiskyShortcutGridworldEnv::get_valid_actions(
        shared_ptr<const mcts::Int3TupleState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        int row = 0;
        int col = 0;
        int time = 0;
        decode_state(state, row, col, time);
        (void)time;

        for (int action = 0; action < num_actions; ++action) {
            if (is_action_valid(row, col, action)) {
                actions->push_back(make_shared<const mcts::IntAction>(action));
            }
        }
        return actions;
    }

    shared_ptr<mcts::Int3TupleStateDistr> RiskyShortcutGridworldEnv::get_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int3TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }

        int row = 0;
        int col = 0;
        int time = 0;
        decode_state(state, row, col, time);

        if (!is_action_valid(row, col, action->action)) {
            throw runtime_error("RiskyShortcutGridworldEnv: invalid action for current state");
        }

        vector<int> legal_actions;
        legal_actions.reserve(num_actions);
        for (int candidate = 0; candidate < num_actions; ++candidate) {
            if (is_action_valid(row, col, candidate)) {
                legal_actions.push_back(candidate);
            }
        }
        if (legal_actions.empty()) {
            throw runtime_error("RiskyShortcutGridworldEnv: non-sink state has no legal actions");
        }

        vector<double> realised_action_probs(num_actions, 0.0);
        const double slip_mass_per_legal_action = slip_prob / static_cast<double>(legal_actions.size());
        for (int legal_action : legal_actions) {
            realised_action_probs[legal_action] = slip_mass_per_legal_action;
        }
        realised_action_probs[action->action] += 1.0 - slip_prob;

        map<tuple<int, int, int>, double> merged;
        const int next_time = time + 1;

        for (int realised_action = 0; realised_action < num_actions; ++realised_action) {
            const double action_prob = realised_action_probs[realised_action];
            if (action_prob <= 0.0) {
                continue;
            }

            auto [base_row, base_col] = apply_action_deterministic(row, col, realised_action);
            if (is_windy_col(base_col)) {
                auto [wind_row, wind_col] = clip(base_row + 1, base_col);
                merged[make_tuple(base_row, base_col, next_time)] += action_prob * (1.0 - wind_prob);
                merged[make_tuple(wind_row, wind_col, next_time)] += action_prob * wind_prob;
            }
            else {
                merged[make_tuple(base_row, base_col, next_time)] += action_prob;
            }
        }

        for (const auto& [key, probability] : merged) {
            const auto& [next_row, next_col, next_time_key] = key;
            distr->emplace(make_state(next_row, next_col, next_time_key), probability);
        }
        return distr;
    }

    shared_ptr<const mcts::Int3TupleState> RiskyShortcutGridworldEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double RiskyShortcutGridworldEnv::get_reward(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const mcts::Int3TupleState> observation) const
    {
        if (is_sink_state(state)) {
            return 0.0;
        }

        auto reward_for_observation = [&](shared_ptr<const mcts::Int3TupleState> next_state) {
            int next_row = 0;
            int next_col = 0;
            int next_time = 0;
            decode_state(next_state, next_row, next_col, next_time);
            (void)next_time;

            if (is_cliff_cell(next_row, next_col)) {
                return cliff_penalty;
            }
            if (is_goal_cell(next_row, next_col)) {
                return step_cost + goal_reward;
            }
            return step_cost;
        };

        if (observation != nullptr) {
            return reward_for_observation(observation);
        }

        double expected_reward = 0.0;
        auto distr = get_transition_distribution(state, action);
        for (const auto& [next_state, probability] : *distr) {
            expected_reward += probability * reward_for_observation(next_state);
        }
        return expected_reward;
    }

    vector<int> RiskyShortcutGridworldEnv::get_windy_cols() const {
        vector<int> cols(windy_cols.begin(), windy_cols.end());
        sort(cols.begin(), cols.end());
        return cols;
    }

    shared_ptr<const mcts::State> RiskyShortcutGridworldEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool RiskyShortcutGridworldEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        return is_sink_state(typed_state);
    }

    shared_ptr<mcts::ActionVector> RiskyShortcutGridworldEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto typed_actions = get_valid_actions(typed_state);
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> RiskyShortcutGridworldEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto typed_action = static_pointer_cast<const mcts::IntAction>(action);
        auto typed_distr = get_transition_distribution(typed_state, typed_action);
        auto distr = make_shared<mcts::StateDistr>();
        for (const auto& [next_state, probability] : *typed_distr) {
            distr->insert_or_assign(static_pointer_cast<const mcts::State>(next_state), probability);
        }
        return distr;
    }

    shared_ptr<const mcts::State> RiskyShortcutGridworldEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto typed_action = static_pointer_cast<const mcts::IntAction>(action);
        auto next_state = sample_transition_distribution(typed_state, typed_action, rand_manager);
        return static_pointer_cast<const mcts::State>(next_state);
    }

    double RiskyShortcutGridworldEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto typed_action = static_pointer_cast<const mcts::IntAction>(action);
        shared_ptr<const mcts::Int3TupleState> typed_observation = nullptr;
        if (observation != nullptr) {
            typed_observation = static_pointer_cast<const mcts::Int3TupleState>(observation);
        }
        return get_reward(typed_state, typed_action, typed_observation);
    }
}
