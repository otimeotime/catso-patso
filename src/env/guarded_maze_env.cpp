#include "env/guarded_maze_env.h"

#include "helper_templates.h"

#include <array>
#include <memory>
#include <stdexcept>

using namespace std;

namespace {
    constexpr array<array<int, mcts::exp::GuardedMazeEnv::num_cols>, mcts::exp::GuardedMazeEnv::num_rows> kMazeMap{{
        {{-1, -1, -1, -1, -1, -1, -1, -1}},
        {{-1,  0,  0,  0,  0,  0,  0, -1}},
        {{-1,  0, -1, -1, -1, -1,  0, -1}},
        {{-1,  0,  0,  0,  0, -1,  0, -1}},
        {{-1,  0,  0,  0,  0, -1,  0, -1}},
        {{-1,  0,  0,  0,  0, -1,  0, -1}},
        {{-1,  0,  0,  0,  0,  0,  0, -1}},
        {{-1, -1, -1, -1, -1, -1, -1, -1}}
    }};

    constexpr array<array<int, 2>, mcts::exp::GuardedMazeEnv::num_actions> kActionDeltas{{
        {{0, 1}},
        {{1, 0}},
        {{0, -1}},
        {{-1, 0}}
    }};

    constexpr array<double, mcts::exp::GuardedMazeEnv::num_guard_reward_outcomes> kGuardRewardValues{{
        -60.0, -30.0, 0.0, 30.0, 60.0
    }};
    constexpr array<double, mcts::exp::GuardedMazeEnv::num_guard_reward_outcomes> kGuardRewardProbabilities{{
        0.06136, 0.24477, 0.38774, 0.24477, 0.06136
    }};
}

namespace mcts::exp {
    GuardedMazeEnv::GuardedMazeEnv(
        array<int, 2> start,
        array<int, 2> goal,
        array<int, 2> guard,
        int max_steps,
        int deduction_limit,
        double reward_normalisation)
        : mcts::MctsEnv(true),
          start(start),
          goal(goal),
          guard(guard),
          max_steps(max_steps),
          deduction_limit(deduction_limit),
          reward_normalisation(reward_normalisation)
    {
        const auto validate_open_cell = [&](const array<int, 2>& cell, const string& name) {
            if (cell[0] < 0 || cell[0] >= num_rows || cell[1] < 0 || cell[1] >= num_cols) {
                throw runtime_error("GuardedMazeEnv: " + name + " is out of bounds");
            }
            if (kMazeMap[cell[0]][cell[1]] == -1) {
                throw runtime_error("GuardedMazeEnv: " + name + " must be on a free cell");
            }
        };

        validate_open_cell(this->start, "start");
        validate_open_cell(this->goal, "goal");
        validate_open_cell(this->guard, "guard");

        if (this->max_steps <= 0) {
            throw runtime_error("GuardedMazeEnv: max_steps must be > 0");
        }
        if (this->deduction_limit < 0) {
            throw runtime_error("GuardedMazeEnv: deduction_limit must be >= 0");
        }
        if (this->reward_normalisation == 0.0) {
            throw runtime_error("GuardedMazeEnv: reward_normalisation must be non-zero");
        }
    }

    bool GuardedMazeEnv::is_wall(int row, int col) const {
        return row < 0 || row >= num_rows || col < 0 || col >= num_cols || kMazeMap[row][col] == -1;
    }

    bool GuardedMazeEnv::is_goal_cell(int row, int col) const {
        return row == goal[0] && col == goal[1];
    }

    bool GuardedMazeEnv::is_guard_cell(int row, int col) const {
        return row == guard[0] && col == guard[1];
    }

    void GuardedMazeEnv::decode_state(
        shared_ptr<const mcts::Int4TupleState> state,
        int& row,
        int& col,
        int& steps,
        int& last_guard_outcome) const
    {
        row = get<0>(state->state);
        col = get<1>(state->state);
        steps = get<2>(state->state);
        last_guard_outcome = get<3>(state->state);
    }

    shared_ptr<const mcts::Int4TupleState> GuardedMazeEnv::make_state(
        int row,
        int col,
        int steps,
        int last_guard_outcome) const
    {
        return make_shared<const mcts::Int4TupleState>(row, col, steps, last_guard_outcome);
    }

    double GuardedMazeEnv::guard_reward_value(int outcome_index) const {
        if (outcome_index < 0 || outcome_index >= num_guard_reward_outcomes) {
            throw runtime_error("GuardedMazeEnv: guard reward outcome index out of range");
        }
        return kGuardRewardValues[static_cast<size_t>(outcome_index)];
    }

    double GuardedMazeEnv::guard_reward_probability(int outcome_index) const {
        if (outcome_index < 0 || outcome_index >= num_guard_reward_outcomes) {
            throw runtime_error("GuardedMazeEnv: guard reward probability index out of range");
        }
        return kGuardRewardProbabilities[static_cast<size_t>(outcome_index)];
    }

    shared_ptr<const mcts::Int4TupleState> GuardedMazeEnv::get_initial_state() const {
        return make_state(start[0], start[1], 0, neutral_guard_reward_outcome);
    }

    bool GuardedMazeEnv::is_sink_state(shared_ptr<const mcts::Int4TupleState> state) const {
        int row = 0;
        int col = 0;
        int steps = 0;
        int last_guard_outcome = 0;
        decode_state(state, row, col, steps, last_guard_outcome);
        (void)last_guard_outcome;
        return is_goal_cell(row, col) || steps >= max_steps;
    }

    shared_ptr<mcts::IntActionVector> GuardedMazeEnv::get_valid_actions(
        shared_ptr<const mcts::Int4TupleState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        int row = 0;
        int col = 0;
        int steps = 0;
        int last_guard_outcome = 0;
        decode_state(state, row, col, steps, last_guard_outcome);
        (void)steps;
        (void)last_guard_outcome;

        for (int action = 0; action < num_actions; ++action) {
            const int next_row = row + kActionDeltas[static_cast<size_t>(action)][0];
            const int next_col = col + kActionDeltas[static_cast<size_t>(action)][1];
            if (!is_wall(next_row, next_col)) {
                actions->push_back(make_shared<const mcts::IntAction>(action));
            }
        }

        return actions;
    }

    shared_ptr<mcts::Int4TupleStateDistr> GuardedMazeEnv::get_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int4TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action < 0 || action->action >= num_actions) {
            throw runtime_error("GuardedMazeEnv: action out of range");
        }

        int row = 0;
        int col = 0;
        int steps = 0;
        int last_guard_outcome = 0;
        decode_state(state, row, col, steps, last_guard_outcome);
        (void)last_guard_outcome;

        const int next_row = row + kActionDeltas[static_cast<size_t>(action->action)][0];
        const int next_col = col + kActionDeltas[static_cast<size_t>(action->action)][1];
        if (is_wall(next_row, next_col)) {
            throw runtime_error("GuardedMazeEnv: invalid action for current state");
        }

        const int next_steps = steps + 1;
        if (is_guard_cell(next_row, next_col)) {
            for (int outcome = 0; outcome < num_guard_reward_outcomes; ++outcome) {
                distr->insert_or_assign(
                    make_state(next_row, next_col, next_steps, outcome),
                    guard_reward_probability(outcome));
            }
        }
        else {
            distr->insert_or_assign(
                make_state(next_row, next_col, next_steps, neutral_guard_reward_outcome),
                1.0);
        }

        return distr;
    }

    shared_ptr<const mcts::Int4TupleState> GuardedMazeEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double GuardedMazeEnv::get_reward(
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
        int last_guard_outcome = 0;
        decode_state(observation, row, col, steps, last_guard_outcome);

        double reward = 0.0;
        if (steps <= deduction_limit) {
            reward -= 1.0;
        }
        if (is_guard_cell(row, col)) {
            reward += guard_reward_value(last_guard_outcome);
        }
        if (is_goal_cell(row, col)) {
            reward += 10.0;
        }

        return reward / reward_normalisation;
    }

    shared_ptr<const mcts::State> GuardedMazeEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool GuardedMazeEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int4TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> GuardedMazeEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int4TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> GuardedMazeEnv::get_transition_distribution_itfc(
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

    shared_ptr<const mcts::State> GuardedMazeEnv::sample_transition_distribution_itfc(
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

    double GuardedMazeEnv::get_reward_itfc(
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
