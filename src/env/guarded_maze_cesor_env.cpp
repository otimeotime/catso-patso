#include "env/guarded_maze_cesor_env.h"

#include "helper_templates.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>

using namespace std;

namespace {
    constexpr array<array<int, 2>, mcts::exp::GuardedMazeCesorEnv::num_actions> kDirectionDeltas{{
        {{-1, 0}},
        {{1, 0}},
        {{0, -1}},
        {{0, 1}}
    }};

    int default_maze_size(int mode) {
        if (mode == 1) {
            return 8;
        }
        if (mode == 2) {
            return 16;
        }
        return 8;
    }

    int default_max_steps(int mode, int rows) {
        if (mode == 1) {
            return 10 * (2 * rows);
        }
        if (mode == 2) {
            return 60 * (2 * rows);
        }
        return 10 * (2 * rows);
    }
}

namespace mcts::exp {
    GuardedMazeCesorEnv::GuardedMazeCesorEnv(
        int mode,
        int rows,
        int cols,
        double action_noise,
        int max_steps,
        double guard_prob,
        double guard_cost,
        double force_motion,
        double reward_normalisation,
        int deduction_limit,
        array<int, 2> start,
        array<int, 2> goal,
        int seed,
        int movement_samples)
        : mcts::MctsEnv(true),
          mode(mode),
          rows(rows > 0 ? rows : default_maze_size(mode)),
          cols(cols > 0 ? cols : (rows > 0 ? rows : default_maze_size(mode))),
          action_noise(action_noise),
          max_steps(max_steps),
          guard_prob(guard_prob),
          guard_cost(guard_cost),
          force_motion(force_motion),
          reward_normalisation(reward_normalisation),
          deduction_limit(deduction_limit),
          L(0),
          start(start),
          goal(goal)
    {
        if (this->rows <= 0 || this->cols <= 0 || this->rows != this->cols) {
            throw runtime_error("GuardedMazeCesorEnv: rows and cols must define a positive square grid");
        }
        if (this->action_noise < 0.0) {
            throw runtime_error("GuardedMazeCesorEnv: action_noise must be >= 0");
        }
        if (this->guard_prob < 0.0 || this->guard_prob > 1.0) {
            throw runtime_error("GuardedMazeCesorEnv: guard_prob must be in [0,1]");
        }
        if (this->guard_cost < 0.0) {
            throw runtime_error("GuardedMazeCesorEnv: guard_cost must be >= 0");
        }
        if (this->reward_normalisation == 0.0) {
            throw runtime_error("GuardedMazeCesorEnv: reward_normalisation must be non-zero");
        }
        if (this->deduction_limit < 0) {
            throw runtime_error("GuardedMazeCesorEnv: deduction_limit must be >= 0");
        }

        L = 2 * this->rows;
        if (this->max_steps <= 0) {
            this->max_steps = default_max_steps(mode, this->rows);
        }

        if (this->goal[0] < 0 || this->goal[1] < 0) {
            this->goal = {this->rows - 2, this->rows / 3};
        }

        build_layout();

        const auto validate_open = [&](const array<int, 2>& cell, const string& name) {
            if (cell[0] < 0 || cell[0] >= this->rows || cell[1] < 0 || cell[1] >= this->cols) {
                throw runtime_error("GuardedMazeCesorEnv: " + name + " is out of bounds");
            }
            if (is_wall(cell[0], cell[1])) {
                throw runtime_error("GuardedMazeCesorEnv: " + name + " must be on a traversable cell");
            }
        };

        validate_open(this->start, "start");
        validate_open(this->goal, "goal");
        precompute_movement_cache(seed, movement_samples);
    }

    int GuardedMazeCesorEnv::encode_tag(bool spotted, int last_event) const {
        if (last_event < normal_event || last_event > spotted_event) {
            throw runtime_error("GuardedMazeCesorEnv: last_event out of range");
        }
        return (spotted ? 1 : 0) * 4 + last_event;
    }

    void GuardedMazeCesorEnv::decode_tag(int tag, bool& spotted, int& last_event) const {
        spotted = (tag / 4) != 0;
        last_event = tag % 4;
        if (last_event < normal_event || last_event > spotted_event) {
            throw runtime_error("GuardedMazeCesorEnv: decoded tag out of range");
        }
    }

    void GuardedMazeCesorEnv::decode_state(
        shared_ptr<const mcts::Int4TupleState> state,
        int& row,
        int& col,
        int& steps,
        bool& spotted,
        int& last_event) const
    {
        row = get<0>(state->state);
        col = get<1>(state->state);
        steps = get<2>(state->state);
        decode_tag(get<3>(state->state), spotted, last_event);
    }

    shared_ptr<const mcts::Int4TupleState> GuardedMazeCesorEnv::make_state(
        int row,
        int col,
        int steps,
        bool spotted,
        int last_event) const
    {
        return make_shared<const mcts::Int4TupleState>(row, col, steps, encode_tag(spotted, last_event));
    }

    bool GuardedMazeCesorEnv::is_wall(int row, int col) const {
        return row < 0 || row >= rows || col < 0 || col >= cols || grid[row][col] == 1;
    }

    bool GuardedMazeCesorEnv::is_goal_cell(int row, int col) const {
        return row == goal[0] && col == goal[1];
    }

    int GuardedMazeCesorEnv::kill_zone_strength(int row, int col) const {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            return 0;
        }
        const int value = grid[row][col];
        return value < 0 ? -value : 0;
    }

    int GuardedMazeCesorEnv::encode_movement_outcome(int row, int col, bool blocked, bool kill_zone) const {
        int code = row;
        code = code * cols + col;
        code = code * 2 + (blocked ? 1 : 0);
        code = code * 2 + (kill_zone ? 1 : 0);
        return code;
    }

    void GuardedMazeCesorEnv::decode_movement_outcome(
        int code,
        int& row,
        int& col,
        bool& blocked,
        bool& kill_zone) const
    {
        kill_zone = (code % 2) != 0;
        code /= 2;
        blocked = (code % 2) != 0;
        code /= 2;
        col = code % cols;
        row = code / cols;
    }

    void GuardedMazeCesorEnv::build_layout() {
        grid.assign(rows, vector<int>(cols, 0));

        for (int row = 0; row < rows; ++row) {
            grid[row][0] = 1;
            grid[row][cols - 1] = 1;
        }
        for (int col = 0; col < cols; ++col) {
            grid[0][col] = 1;
            grid[rows - 1][col] = 1;
        }

        if (mode == 1) {
            for (int row = cols / 4 + 1; row < (3 * cols) / 4; ++row) {
                grid[row][(3 * rows) / 4 - 1] = 1;
            }
            for (int col = rows / 4; col < (3 * rows) / 4; ++col) {
                grid[(3 * cols) / 4 - 1][col] = 1;
            }
            if (guard_prob > 0.0) {
                for (int col = 1; col < rows / 4; ++col) {
                    grid[(3 * cols) / 4 - 1][col] = -1;
                }
            }
        }
        else if (mode == 2) {
            for (int row = 0; row < cols / 2 - 1; ++row) {
                for (int col = rows / 2 - 1; col < rows / 2 + 1; ++col) {
                    grid[row][col] = 1;
                }
            }
            for (int row = cols / 4 + 1; row < (3 * cols) / 4 + 1; ++row) {
                for (int col = (3 * rows) / 4 - 1; col < (3 * rows) / 4 + 1; ++col) {
                    grid[row][col] = 1;
                }
            }
            for (int row = (3 * cols) / 4 - 1; row < (3 * cols) / 4 + 1; ++row) {
                for (int col = rows / 4; col < (3 * rows) / 4; ++col) {
                    grid[row][col] = 1;
                }
            }
            if (guard_prob > 0.0) {
                for (int row = (3 * cols) / 4 - 1; row < (3 * cols) / 4 + 1; ++row) {
                    for (int col = 1; col < rows / 4; ++col) {
                        grid[row][col] = -1;
                    }
                }
            }
        }
    }

    void GuardedMazeCesorEnv::precompute_movement_cache(int seed, int samples_per_pair) {
        movement_cache.assign(static_cast<size_t>(rows * cols * num_actions), {});
        mt19937 rng(static_cast<uint32_t>(seed));
        normal_distribution<double> noise(0.0, action_noise);

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (is_wall(row, col)) {
                    continue;
                }

                for (int action = 0; action < num_actions; ++action) {
                    unordered_map<int, double> counts;
                    const double current_x = static_cast<double>(row);
                    const double current_y = static_cast<double>(col);
                    const double target_x = current_x + kDirectionDeltas[static_cast<size_t>(action)][0];
                    const double target_y = current_y + kDirectionDeltas[static_cast<size_t>(action)][1];

                    for (int sample = 0; sample < max(samples_per_pair, 1); ++sample) {
                        const double noisy_x = clamp(target_x + noise(rng), 0.0, static_cast<double>(rows - 1));
                        const double noisy_y = clamp(target_y + noise(rng), 0.0, static_cast<double>(cols - 1));

                        const int next_row_raw = static_cast<int>(lround(noisy_x));
                        const int next_col_raw = static_cast<int>(lround(noisy_y));
                        const int mid_row = static_cast<int>(lround(0.5 * (current_x + noisy_x)));
                        const int mid_col = static_cast<int>(lround(0.5 * (current_y + noisy_y)));

                        bool blocked = is_wall(next_row_raw, next_col_raw) || is_wall(mid_row, mid_col);
                        int next_row = next_row_raw;
                        int next_col = next_col_raw;
                        if (blocked) {
                            next_row = row;
                            next_col = col;
                        }

                        const bool kill_zone =
                            max(kill_zone_strength(next_row, next_col), kill_zone_strength(mid_row, mid_col)) > 0;
                        counts[encode_movement_outcome(next_row, next_col, blocked, kill_zone)] += 1.0;
                    }

                    auto& probs = movement_cache[static_cast<size_t>((row * cols + col) * num_actions + action)];
                    for (const auto& [code, count] : counts) {
                        probs[code] = count / static_cast<double>(max(samples_per_pair, 1));
                    }
                }
            }
        }
    }

    shared_ptr<const mcts::Int4TupleState> GuardedMazeCesorEnv::get_initial_state() const {
        return make_state(start[0], start[1], 0, false, normal_event);
    }

    bool GuardedMazeCesorEnv::is_sink_state(shared_ptr<const mcts::Int4TupleState> state) const {
        int row = 0;
        int col = 0;
        int steps = 0;
        bool spotted = false;
        int last_event = normal_event;
        decode_state(state, row, col, steps, spotted, last_event);
        (void)spotted;
        (void)last_event;
        return is_goal_cell(row, col) || steps >= max_steps;
    }

    bool GuardedMazeCesorEnv::is_catastrophic_state(shared_ptr<const mcts::Int4TupleState> state) const {
        int row = 0;
        int col = 0;
        int steps = 0;
        bool spotted = false;
        int last_event = normal_event;
        decode_state(state, row, col, steps, spotted, last_event);
        (void)row;
        (void)col;
        (void)steps;
        return last_event == spotted_event;
    }

    shared_ptr<mcts::IntActionVector> GuardedMazeCesorEnv::get_valid_actions(
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

    shared_ptr<mcts::Int4TupleStateDistr> GuardedMazeCesorEnv::get_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int4TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action < 0 || action->action >= num_actions) {
            throw runtime_error("GuardedMazeCesorEnv: action out of range");
        }

        int row = 0;
        int col = 0;
        int steps = 0;
        bool spotted = false;
        int last_event = normal_event;
        decode_state(state, row, col, steps, spotted, last_event);

        const auto& base_probs = movement_cache[static_cast<size_t>((row * cols + col) * num_actions + action->action)];
        const int next_steps = steps + 1;

        for (const auto& [code, probability] : base_probs) {
            int next_row = 0;
            int next_col = 0;
            bool blocked = false;
            bool kill_zone = false;
            decode_movement_outcome(code, next_row, next_col, blocked, kill_zone);

            if (!spotted && kill_zone && guard_prob > 0.0) {
                distr->insert_or_assign(
                    make_state(next_row, next_col, next_steps, false, blocked ? blocked_event : normal_event),
                    probability * (1.0 - guard_prob));
                distr->insert_or_assign(
                    make_state(next_row, next_col, next_steps, true, spotted_event),
                    probability * guard_prob);
            }
            else {
                distr->insert_or_assign(
                    make_state(next_row, next_col, next_steps, spotted, blocked ? blocked_event : normal_event),
                    probability);
            }
        }

        return distr;
    }

    shared_ptr<const mcts::Int4TupleState> GuardedMazeCesorEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double GuardedMazeCesorEnv::get_reward(
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
        bool spotted = false;
        int last_event = normal_event;
        decode_state(state, row, col, steps, spotted, last_event);

        int next_row = 0;
        int next_col = 0;
        int next_steps = 0;
        bool next_spotted = false;
        int next_last_event = normal_event;
        decode_state(observation, next_row, next_col, next_steps, next_spotted, next_last_event);
        (void)next_spotted;
        (void)next_steps;

        double reward = 0.0;
        if (is_goal_cell(next_row, next_col)) {
            reward += static_cast<double>(L);
        }
        else if (steps < deduction_limit) {
            reward -= 1.0;
        }

        if (force_motion > 0.0 && next_last_event == blocked_event) {
            reward -= force_motion;
        }
        if (next_last_event == spotted_event) {
            reward -= guard_cost * static_cast<double>(L);
        }

        return reward / reward_normalisation;
    }

    shared_ptr<const mcts::State> GuardedMazeCesorEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool GuardedMazeCesorEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int4TupleState>(state));
    }

    bool GuardedMazeCesorEnv::is_catastrophic_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_catastrophic_state(static_pointer_cast<const mcts::Int4TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> GuardedMazeCesorEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int4TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> GuardedMazeCesorEnv::get_transition_distribution_itfc(
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

    shared_ptr<const mcts::State> GuardedMazeCesorEnv::sample_transition_distribution_itfc(
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

    double GuardedMazeCesorEnv::get_reward_itfc(
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
