#include "env/laser_tag_safe_grid_env.h"
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
    LaserTagSafeGridEnv::LaserTagSafeGridEnv(
        int N,
        double obstacles_frac,
        double eps_shot,
        double step_cost,
        double goal_reward,
        double shot_penalty,
        double slip_prob,
        int max_steps,
        int seed)
        : mcts::MctsEnv(true),
          N(N),
          eps_shot(eps_shot),
          step_cost(step_cost),
          goal_reward(goal_reward),
          shot_penalty(shot_penalty),
          slip_prob(slip_prob),
          max_steps(max_steps),
          obstacles()
    {
        if (this->N < 5) {
            throw runtime_error("LaserTagSafeGridEnv: N must be >= 5");
        }
        if (obstacles_frac < 0.0 || obstacles_frac > 1.0) {
            throw runtime_error("LaserTagSafeGridEnv: obstacles_frac must be in [0,1]");
        }
        if (this->eps_shot < 0.0 || this->eps_shot > 1.0) {
            throw runtime_error("LaserTagSafeGridEnv: eps_shot must be in [0,1]");
        }
        if (this->slip_prob < 0.0 || this->slip_prob > 1.0) {
            throw runtime_error("LaserTagSafeGridEnv: slip_prob must be in [0,1]");
        }
        if (this->max_steps <= 0) {
            throw runtime_error("LaserTagSafeGridEnv: max_steps must be > 0");
        }

        const int sentry_row = this->N / 2;
        const int sentry_col = this->N / 2;
        vector<int> cells;
        cells.reserve(this->N * this->N - 3);
        for (int row = 0; row < this->N; ++row) {
            for (int col = 0; col < this->N; ++col) {
                const bool is_start = (row == this->N - 1 && col == 0);
                const bool is_goal = (row == 0 && col == this->N - 1);
                const bool is_sentry = (row == sentry_row && col == sentry_col);
                if (!is_start && !is_goal && !is_sentry) {
                    cells.push_back(encode_cell(row, col));
                }
            }
        }

        int num_obstacles = static_cast<int>(round(obstacles_frac * static_cast<double>(cells.size())));
        num_obstacles = max(0, min(num_obstacles, static_cast<int>(cells.size())));

        auto rng = detail::make_seeded_rng(seed);
        shuffle(cells.begin(), cells.end(), rng);
        for (int i = 0; i < num_obstacles; ++i) {
            obstacles.insert(cells[i]);
        }
    }

    int LaserTagSafeGridEnv::encode_cell(int row, int col) const {
        return row * N + col;
    }

    bool LaserTagSafeGridEnv::is_obstacle(int row, int col) const {
        return obstacles.find(encode_cell(row, col)) != obstacles.end();
    }

    bool LaserTagSafeGridEnv::is_free(int row, int col) const {
        return row >= 0 && row < N && col >= 0 && col < N && !is_obstacle(row, col);
    }

    pair<int, int> LaserTagSafeGridEnv::step_to(int row, int col, int action) const {
        static const array<pair<int, int>, num_actions> deltas = {{
            {-1, 0}, {0, 1}, {1, 0}, {0, -1}
        }};
        const int next_row = row + deltas[action].first;
        const int next_col = col + deltas[action].second;
        if (!is_free(next_row, next_col)) {
            return {row, col};
        }
        return {next_row, next_col};
    }

    bool LaserTagSafeGridEnv::is_exposed(int row, int col) const {
        const int sentry_row = N / 2;
        const int sentry_col = N / 2;

        if (row == sentry_row) {
            const int step = (col < sentry_col) ? 1 : -1;
            for (int c = col + step; c != sentry_col; c += step) {
                if (is_obstacle(row, c)) {
                    return false;
                }
            }
            return true;
        }

        if (col == sentry_col) {
            const int step = (row < sentry_row) ? 1 : -1;
            for (int r = row + step; r != sentry_row; r += step) {
                if (is_obstacle(r, col)) {
                    return false;
                }
            }
            return true;
        }

        return false;
    }

    void LaserTagSafeGridEnv::decode_state(
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

    shared_ptr<const mcts::Int4TupleState> LaserTagSafeGridEnv::make_state(
        int row,
        int col,
        int steps,
        int status) const
    {
        return make_shared<const mcts::Int4TupleState>(row, col, steps, status);
    }

    shared_ptr<const mcts::Int4TupleState> LaserTagSafeGridEnv::get_initial_state() const {
        return make_state(N - 1, 0, 0, active_status);
    }

    bool LaserTagSafeGridEnv::is_sink_state(shared_ptr<const mcts::Int4TupleState> state) const {
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

    shared_ptr<mcts::IntActionVector> LaserTagSafeGridEnv::get_valid_actions(
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

    shared_ptr<mcts::Int4TupleStateDistr> LaserTagSafeGridEnv::get_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int4TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action < 0 || action->action >= num_actions) {
            throw runtime_error("LaserTagSafeGridEnv: action out of range");
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

        for (int actual_action = 0; actual_action < num_actions; ++actual_action) {
            const double action_prob = actual_action_probs[actual_action];
            if (action_prob <= 0.0) {
                continue;
            }

            const auto [next_row, next_col] = step_to(row, col, actual_action);
            if (is_exposed(next_row, next_col)) {
                detail::accumulate_probability(
                    distr,
                    make_state(next_row, next_col, next_steps, shot_status),
                    action_prob * eps_shot);
                const bool is_goal = (next_row == 0 && next_col == N - 1);
                const int safe_status = is_goal ? goal_status : ((next_steps >= max_steps) ? truncated_status : active_status);
                detail::accumulate_probability(
                    distr,
                    make_state(next_row, next_col, next_steps, safe_status),
                    action_prob * (1.0 - eps_shot));
                continue;
            }

            const int next_status = (next_row == 0 && next_col == N - 1)
                ? goal_status
                : ((next_steps >= max_steps) ? truncated_status : active_status);
            detail::accumulate_probability(distr, make_state(next_row, next_col, next_steps, next_status), action_prob);
        }

        return distr;
    }

    shared_ptr<const mcts::Int4TupleState> LaserTagSafeGridEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double LaserTagSafeGridEnv::get_reward(
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

        if (status == shot_status) {
            return shot_penalty;
        }
        if (status == goal_status) {
            return step_cost + goal_reward;
        }
        return step_cost;
    }

    shared_ptr<const mcts::State> LaserTagSafeGridEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool LaserTagSafeGridEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int4TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> LaserTagSafeGridEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int4TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> LaserTagSafeGridEnv::get_transition_distribution_itfc(
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

    shared_ptr<const mcts::State> LaserTagSafeGridEnv::sample_transition_distribution_itfc(
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

    double LaserTagSafeGridEnv::get_reward_itfc(
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