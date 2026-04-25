#include "env/thin_ice_frozen_lake_plus_env.h"

#include "env/discrete_env_utils.h"
#include "helper_templates.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace std;

namespace mcts::exp {
    ThinIceFrozenLakePlusEnv::ThinIceFrozenLakePlusEnv(
        int N,
        double slip_prob,
        double thin_frac,
        double break_prob,
        double step_cost,
        double goal_reward,
        double break_penalty,
        int max_steps,
        int seed)
        : mcts::MctsEnv(true),
          N(N),
          slip_prob(slip_prob),
          break_prob(break_prob),
          step_cost(step_cost),
          goal_reward(goal_reward),
          break_penalty(break_penalty),
          max_steps(max_steps > 0 ? max_steps : 10 * N * N),
          thin_cells()
    {
        if (this->N < 4) {
            throw runtime_error("ThinIceFrozenLakePlusEnv: N must be >= 4");
        }
        if (this->slip_prob < 0.0 || this->slip_prob > 1.0) {
            throw runtime_error("ThinIceFrozenLakePlusEnv: slip_prob must be in [0,1]");
        }
        if (thin_frac < 0.0 || thin_frac > 1.0) {
            throw runtime_error("ThinIceFrozenLakePlusEnv: thin_frac must be in [0,1]");
        }
        if (this->break_prob < 0.0 || this->break_prob > 1.0) {
            throw runtime_error("ThinIceFrozenLakePlusEnv: break_prob must be in [0,1]");
        }
        if (this->max_steps <= 0) {
            throw runtime_error("ThinIceFrozenLakePlusEnv: max_steps must be > 0");
        }

        vector<int> candidate_cells;
        candidate_cells.reserve(this->N * this->N - 2);
        for (int row = 0; row < this->N; ++row) {
            for (int col = 0; col < this->N; ++col) {
                const bool is_start = (row == 0 && col == 0);
                const bool is_goal = (row == this->N - 1 && col == this->N - 1);
                if (!is_start && !is_goal) {
                    candidate_cells.push_back(encode_cell(row, col));
                }
            }
        }

        int num_thin = static_cast<int>(round(thin_frac * static_cast<double>(candidate_cells.size())));
        num_thin = max(1, num_thin);
        num_thin = min(num_thin, static_cast<int>(candidate_cells.size()));

        auto rng = detail::make_seeded_rng(seed);
        shuffle(candidate_cells.begin(), candidate_cells.end(), rng);
        for (int i = 0; i < num_thin; ++i) {
            thin_cells.insert(candidate_cells[i]);
        }
    }

    int ThinIceFrozenLakePlusEnv::encode_cell(int row, int col) const {
        return row * N + col;
    }

    bool ThinIceFrozenLakePlusEnv::is_thin(int row, int col) const {
        return thin_cells.find(encode_cell(row, col)) != thin_cells.end();
    }

    pair<int, int> ThinIceFrozenLakePlusEnv::clip(int row, int col) const {
        return {clamp(row, 0, N - 1), clamp(col, 0, N - 1)};
    }

    void ThinIceFrozenLakePlusEnv::decode_state(
        shared_ptr<const mcts::Int4TupleState> state,
        int& row,
        int& col,
        int& steps,
        int& status) const
    {
        row = get<0>(state->state);
        col = get<1>(state->state);
        steps = get<2>(state->state);
        status = get<3>(state->state);
    }

    shared_ptr<const mcts::Int4TupleState> ThinIceFrozenLakePlusEnv::make_state(
        int row,
        int col,
        int steps,
        int status) const
    {
        return make_shared<const mcts::Int4TupleState>(row, col, steps, status);
    }

    shared_ptr<const mcts::Int4TupleState> ThinIceFrozenLakePlusEnv::get_initial_state() const {
        return make_state(0, 0, 0, active_status);
    }

    bool ThinIceFrozenLakePlusEnv::is_sink_state(shared_ptr<const mcts::Int4TupleState> state) const {
        int row = 0;
        int col = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, row, col, steps, status);
        (void)row;
        (void)col;
        (void)steps;
        return status != active_status;
    }

    shared_ptr<mcts::IntActionVector> ThinIceFrozenLakePlusEnv::get_valid_actions(
        shared_ptr<const mcts::Int4TupleState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        for (int action = 0; action < num_actions; ++action) {
            actions->push_back(make_shared<const mcts::IntAction>(action));
        }
        return actions;
    }

    shared_ptr<mcts::Int4TupleStateDistr> ThinIceFrozenLakePlusEnv::get_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int4TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action < 0 || action->action >= num_actions) {
            throw runtime_error("ThinIceFrozenLakePlusEnv: action out of range");
        }

        int row = 0;
        int col = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, row, col, steps, status);
        (void)status;
        const int next_steps = steps + 1;

        array<double, num_actions> actual_action_probs{};
        actual_action_probs.fill(slip_prob / static_cast<double>(num_actions));
        actual_action_probs[action->action] += 1.0 - slip_prob;

        static const array<pair<int, int>, num_actions> deltas = {{
            {-1, 0}, {0, 1}, {1, 0}, {0, -1}
        }};

        for (int actual_action = 0; actual_action < num_actions; ++actual_action) {
            const double action_prob = actual_action_probs[actual_action];
            if (action_prob <= 0.0) {
                continue;
            }

            const auto [next_row, next_col] = clip(row + deltas[actual_action].first, col + deltas[actual_action].second);
            if (next_row == N - 1 && next_col == N - 1) {
                detail::accumulate_probability(
                    distr,
                    make_state(next_row, next_col, next_steps, goal_status),
                    action_prob);
                continue;
            }

            if (is_thin(next_row, next_col)) {
                detail::accumulate_probability(
                    distr,
                    make_state(next_row, next_col, next_steps, broken_status),
                    action_prob * break_prob);
                const int safe_status = (next_steps >= max_steps) ? truncated_status : active_status;
                detail::accumulate_probability(
                    distr,
                    make_state(next_row, next_col, next_steps, safe_status),
                    action_prob * (1.0 - break_prob));
                continue;
            }

            const int next_status = (next_steps >= max_steps) ? truncated_status : active_status;
            detail::accumulate_probability(distr, make_state(next_row, next_col, next_steps, next_status), action_prob);
        }

        return distr;
    }

    shared_ptr<const mcts::Int4TupleState> ThinIceFrozenLakePlusEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double ThinIceFrozenLakePlusEnv::get_reward(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const mcts::Int4TupleState> observation) const
    {
        if (is_sink_state(state)) {
            return 0.0;
        }

        if (observation == nullptr) {
            double expected_reward = 0.0;
            auto distr = get_transition_distribution(state, action);
            for (const auto& [next_state, probability] : *distr) {
                expected_reward += probability * get_reward(state, action, next_state);
            }
            return expected_reward;
        }

        int row = 0;
        int col = 0;
        int steps = 0;
        int status = 0;
        decode_state(observation, row, col, steps, status);
        (void)row;
        (void)col;
        (void)steps;

        if (status == goal_status) {
            return step_cost + goal_reward;
        }
        if (status == broken_status) {
            return break_penalty;
        }
        return step_cost;
    }

    shared_ptr<const mcts::State> ThinIceFrozenLakePlusEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool ThinIceFrozenLakePlusEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int4TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> ThinIceFrozenLakePlusEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int4TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> ThinIceFrozenLakePlusEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto typed_distr = get_transition_distribution(
            static_pointer_cast<const mcts::Int4TupleState>(state),
            static_pointer_cast<const mcts::IntAction>(action));
        auto distr = make_shared<mcts::StateDistr>();
        for (const auto& [next_state, probability] : *typed_distr) {
            distr->insert_or_assign(static_pointer_cast<const mcts::State>(next_state), probability);
        }
        return distr;
    }

    shared_ptr<const mcts::State> ThinIceFrozenLakePlusEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        return static_pointer_cast<const mcts::State>(
            sample_transition_distribution(
                static_pointer_cast<const mcts::Int4TupleState>(state),
                static_pointer_cast<const mcts::IntAction>(action),
                rand_manager));
    }

    double ThinIceFrozenLakePlusEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        shared_ptr<const mcts::Int4TupleState> typed_observation = nullptr;
        if (observation != nullptr) {
            typed_observation = static_pointer_cast<const mcts::Int4TupleState>(observation);
        }
        return get_reward(
            static_pointer_cast<const mcts::Int4TupleState>(state),
            static_pointer_cast<const mcts::IntAction>(action),
            typed_observation);
    }
}
