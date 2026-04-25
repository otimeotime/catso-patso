#include "env/two_path_ssp_deceptive_env.h"

#include "helper_templates.h"

#include <memory>
#include <stdexcept>

using namespace std;

namespace mcts::exp {
    TwoPathSspDeceptiveEnv::TwoPathSspDeceptiveEnv(
        int safe_length,
        int risky_length,
        int hazard_index,
        double hazard_prob,
        double step_cost,
        double goal_reward,
        double hazard_penalty,
        int max_steps)
        : mcts::MctsEnv(true),
          safe_length(safe_length),
          risky_length(risky_length),
          hazard_index(hazard_index),
          hazard_prob(hazard_prob),
          step_cost(step_cost),
          goal_reward(goal_reward),
          hazard_penalty(hazard_penalty),
          max_steps(max_steps)
    {
        if (this->safe_length < 2 || this->risky_length < 1) {
            throw runtime_error("TwoPathSspDeceptiveEnv: invalid path lengths");
        }
        if (this->hazard_prob < 0.0 || this->hazard_prob > 1.0) {
            throw runtime_error("TwoPathSspDeceptiveEnv: hazard_prob must be in [0,1]");
        }
        if (this->max_steps <= 0) {
            throw runtime_error("TwoPathSspDeceptiveEnv: max_steps must be > 0");
        }
    }

    void TwoPathSspDeceptiveEnv::decode_state(
        shared_ptr<const mcts::Int4TupleState> state,
        int& mode,
        int& index,
        int& steps,
        int& status) const
    {
        mode = get<0>(state->state);
        index = get<1>(state->state);
        steps = get<2>(state->state);
        status = get<3>(state->state);
    }

    shared_ptr<const mcts::Int4TupleState> TwoPathSspDeceptiveEnv::make_state(
        int mode,
        int index,
        int steps,
        int status) const
    {
        return make_shared<const mcts::Int4TupleState>(mode, index, steps, status);
    }

    shared_ptr<const mcts::Int4TupleState> TwoPathSspDeceptiveEnv::get_initial_state() const {
        return make_state(root_mode, 0, 0, active_status);
    }

    bool TwoPathSspDeceptiveEnv::is_sink_state(shared_ptr<const mcts::Int4TupleState> state) const {
        int mode = 0;
        int index = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, mode, index, steps, status);
        (void)mode;
        (void)index;
        (void)steps;
        return status != active_status;
    }

    shared_ptr<mcts::IntActionVector> TwoPathSspDeceptiveEnv::get_valid_actions(
        shared_ptr<const mcts::Int4TupleState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        int mode = 0;
        int index = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, mode, index, steps, status);
        (void)index;
        (void)steps;
        (void)status;

        if (mode == root_mode) {
            actions->push_back(make_shared<const mcts::IntAction>(choose_safe_action));
            actions->push_back(make_shared<const mcts::IntAction>(choose_risky_action));
        } else {
            actions->push_back(make_shared<const mcts::IntAction>(continue_action));
        }
        return actions;
    }

    shared_ptr<mcts::Int4TupleStateDistr> TwoPathSspDeceptiveEnv::get_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int4TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }

        int mode = 0;
        int index = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, mode, index, steps, status);
        (void)status;
        const int next_steps = steps + 1;

        if (mode == root_mode) {
            if (action->action != choose_safe_action && action->action != choose_risky_action) {
                throw runtime_error("TwoPathSspDeceptiveEnv: invalid root action");
            }
            const int next_mode = (action->action == choose_safe_action) ? safe_mode : risky_mode;
            const int next_status = (next_steps >= max_steps) ? truncated_status : active_status;
            distr->insert_or_assign(make_state(next_mode, 0, next_steps, next_status), 1.0);
            return distr;
        }

        if (action->action != continue_action) {
            throw runtime_error("TwoPathSspDeceptiveEnv: only continue action is valid after path choice");
        }

        if (mode == safe_mode) {
            if (index + 1 >= safe_length) {
                distr->insert_or_assign(make_state(mode, index, next_steps, goal_status), 1.0);
            } else {
                const int next_status = (next_steps >= max_steps) ? truncated_status : active_status;
                distr->insert_or_assign(make_state(mode, index + 1, next_steps, next_status), 1.0);
            }
            return distr;
        }

        if (index == hazard_index) {
            distr->insert_or_assign(make_state(mode, index, next_steps, hazard_status), hazard_prob);
            if (index + 1 >= risky_length) {
                distr->insert_or_assign(make_state(mode, index, next_steps, goal_status), 1.0 - hazard_prob);
            } else {
                const int next_status = (next_steps >= max_steps) ? truncated_status : active_status;
                distr->insert_or_assign(make_state(mode, index + 1, next_steps, next_status), 1.0 - hazard_prob);
            }
            return distr;
        }

        if (index + 1 >= risky_length) {
            distr->insert_or_assign(make_state(mode, index, next_steps, goal_status), 1.0);
        } else {
            const int next_status = (next_steps >= max_steps) ? truncated_status : active_status;
            distr->insert_or_assign(make_state(mode, index + 1, next_steps, next_status), 1.0);
        }
        return distr;
    }

    shared_ptr<const mcts::Int4TupleState> TwoPathSspDeceptiveEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double TwoPathSspDeceptiveEnv::get_reward(
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

        int mode = 0;
        int index = 0;
        int steps = 0;
        int status = 0;
        decode_state(observation, mode, index, steps, status);
        (void)mode;
        (void)index;
        (void)steps;

        if (status == hazard_status) {
            return hazard_penalty;
        }
        if (status == goal_status) {
            return step_cost + goal_reward;
        }
        return step_cost;
    }

    shared_ptr<const mcts::State> TwoPathSspDeceptiveEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool TwoPathSspDeceptiveEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int4TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> TwoPathSspDeceptiveEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int4TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> TwoPathSspDeceptiveEnv::get_transition_distribution_itfc(
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

    shared_ptr<const mcts::State> TwoPathSspDeceptiveEnv::sample_transition_distribution_itfc(
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

    double TwoPathSspDeceptiveEnv::get_reward_itfc(
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
